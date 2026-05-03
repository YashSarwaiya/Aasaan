"""Universal multi-task training curriculum.

Six task SHAPES that work on any domain — medical, legal, logs, support
tickets, sales notes, code, anything. None of these tasks hardcode field
names or domain language; the auto-detected `domain` string is plugged into
each instruction at generation time.

Task shapes:
  1. EXTRACT     — raw text → structured JSON (uses whatever schema we built)
  2. SUMMARIZE   — text → 1-3 sentences (skips very short docs)
  3. QA          — text + question → answer
  4. PARAPHRASE  — text → same content, different words
  5. YES_NO      — text + binary question → Yes/No
  6. REFUSE      — unanswerable question + text → "Not in the document"

Tasks 2-6 use the teacher LLM. Task 1 is self-generating from the structured
extraction we already did.

Each row: {instruction, input, output, task} — Alpaca format ready for SFT.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Any, Callable

from .llm import batch_ask
from .qa import has_content_for_question, is_empty_value


# ── Helpers ──────────────────────────────────────────────────────────────


def _short_doc(text: str, min_chars: int = 200) -> bool:
    """Skip tasks that don't make sense on very short inputs (e.g. 1-line logs)."""
    return len(text) < min_chars


# ── Task 1: EXTRACT — raw text → structured JSON ─────────────────────────


def task_extract(items: list[dict[str, Any]], domain: str) -> list[dict[str, str]]:
    """Self-generating: input = raw text, output = the structured form we already pulled.

    No LLM calls — we already paid that cost during extraction. This is the
    cheapest, most universally-useful task.
    """
    rows: list[dict[str, str]] = []
    instruction = f"Extract the structured fields from this {domain} document as JSON."
    for item in items:
        if is_empty_value(item.get("structured")):
            continue
        rows.append({
            "task": "extract",
            "instruction": instruction,
            "input": item["original"][:2500],
            "output": json.dumps(item["structured"], indent=2)[:1500],
        })
    return rows


# ── Task 2: SUMMARIZE — text → short summary ─────────────────────────────


def make_summary_prompt(domain: str, text: str) -> str:
    return f"""Summarize this {domain} document in 2-3 sentences. Keep all key
facts intact. No embellishment, no preamble — just the summary.

Document:
{text[:2500]}

Summary:"""


def task_summarize(
    teacher_model, teacher_tok, items: list[dict[str, Any]], domain: str,
    *, batch_size: int = 8,
) -> list[dict[str, str]]:
    """Teacher-generated. Skips very short docs (logs, single-line errors)."""
    candidates = [it for it in items if not _short_doc(it.get("original", ""))]
    if not candidates:
        return []

    prompts = [make_summary_prompt(domain, it["original"]) for it in candidates]
    instruction = f"Summarize this {domain} document in 2-3 sentences."

    rows: list[dict[str, str]] = []
    for start in range(0, len(prompts), batch_size):
        batch_items = candidates[start:start + batch_size]
        batch_prompts = prompts[start:start + batch_size]
        outputs = batch_ask(teacher_model, teacher_tok, batch_prompts, max_tokens=300)
        for item, out in zip(batch_items, outputs):
            if len(out.strip()) < 50:
                continue
            rows.append({
                "task": "summarize",
                "instruction": instruction,
                "input": item["original"][:2500],
                "output": out.strip()[:600],
            })
    return rows


# ── Task 3: QA — text + question → answer ────────────────────────────────


def make_qa_prompt(domain: str, structured: dict[str, Any], question: str) -> str:
    context = json.dumps(structured, indent=2)[:1500]
    return f"""Domain: {domain}

Structured data extracted from the document:
{context}

Question: {question}

Give a concise direct answer (1-3 sentences). Only state facts grounded in the
data above. If the info is missing, say "Not specified in the document."

Answer:"""


def task_qa(
    teacher_model, teacher_tok,
    items: list[dict[str, Any]],
    questions: list[str],
    domain: str,
    *,
    questions_per_doc: int = 4,
    batch_size: int = 16,
) -> list[dict[str, str]]:
    """Teacher-generated Q&A pairs over schema-derived questions."""
    if not questions:
        return []

    pairs: list[tuple[dict[str, Any], str, str]] = []
    for item in items:
        candidate = [q for q in questions if has_content_for_question(item.get("structured", {}), q)]
        if not candidate:
            continue
        for q in candidate[:questions_per_doc]:
            pairs.append((item, q, make_qa_prompt(domain, item["structured"], q)))

    rows: list[dict[str, str]] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        prompts = [p[2] for p in batch]
        outputs = batch_ask(teacher_model, teacher_tok, prompts, max_tokens=200)
        for (item, q, _), out in zip(batch, outputs):
            if len(out.strip()) < 20:
                continue
            rows.append({
                "task": "qa",
                "instruction": q,
                "input": item["original"][:2500],
                "output": out.strip()[:500],
            })
    return rows


# ── Task 4: PARAPHRASE — text → same content, different words ────────────


def make_paraphrase_prompt(domain: str, text: str) -> str:
    return f"""Rephrase this {domain} document. Use different sentence structures
and synonyms. Keep ALL facts identical (same names, numbers, dates, IDs,
identifiers, quoted text). Maintain the original tone.

Original:
{text[:2000]}

Paraphrased:"""


def task_paraphrase(
    teacher_model, teacher_tok, items: list[dict[str, Any]], domain: str,
    *, batch_size: int = 8, max_examples: int = 200,
) -> list[dict[str, str]]:
    """Teacher-generated paraphrases. Skipped on short structured docs (logs)
    where paraphrasing would damage exact-text extraction skills."""
    candidates = [it for it in items if not _short_doc(it.get("original", ""))]
    candidates = candidates[:max_examples]
    if not candidates:
        return []

    prompts = [make_paraphrase_prompt(domain, it["original"]) for it in candidates]
    instruction = f"Rewrite this {domain} document in different words while keeping all facts identical."

    rows: list[dict[str, str]] = []
    for start in range(0, len(prompts), batch_size):
        batch_items = candidates[start:start + batch_size]
        batch_prompts = prompts[start:start + batch_size]
        outputs = batch_ask(teacher_model, teacher_tok, batch_prompts, max_tokens=400)
        for item, out in zip(batch_items, outputs):
            if len(out.strip()) < 100:
                continue
            rows.append({
                "task": "paraphrase",
                "instruction": instruction,
                "input": item["original"][:2500],
                "output": out.strip()[:1500],
            })
    return rows


# ── Task 5: YES_NO — bounded yes/no questions grounded in source ─────────


def make_yes_no_prompt(domain: str, text: str) -> str:
    return f"""You are creating training data for a {domain} assistant.

Given this document, write 2 yes/no questions a user might ask, with their
correct answers. Questions must have unambiguous yes/no answers grounded
strictly in the document's content.

Document:
{text[:1800]}

Output exactly 2 lines, each in this format:
Q: <question> | A: <Yes or No>

Lines:"""


def task_yes_no(
    teacher_model, teacher_tok, items: list[dict[str, Any]], domain: str,
    *, batch_size: int = 8,
) -> list[dict[str, str]]:
    """Teacher-generated yes/no Q&A. Trains the model to give bounded answers."""
    if not items:
        return []
    prompts = [make_yes_no_prompt(domain, it["original"]) for it in items]

    rows: list[dict[str, str]] = []
    for start in range(0, len(prompts), batch_size):
        batch_items = items[start:start + batch_size]
        batch_prompts = prompts[start:start + batch_size]
        outputs = batch_ask(teacher_model, teacher_tok, batch_prompts, max_tokens=200)
        for item, out in zip(batch_items, outputs):
            for line in out.split("\n"):
                line = line.strip()
                if "|" not in line:
                    continue
                q_part, _, a_part = line.partition("|")
                q = q_part.replace("Q:", "").strip()
                a = a_part.replace("A:", "").strip()
                if not q or a.lower() not in ("yes", "no"):
                    continue
                rows.append({
                    "task": "yes_no",
                    "instruction": q[:200],
                    "input": item["original"][:2500],
                    "output": a.capitalize(),
                })
    return rows


# ── Task 6: REFUSE — confidently say "I don't know" for out-of-doc questions ─


def make_refuse_prompt(domain: str, text: str) -> str:
    return f"""Generate ONE question that CANNOT be answered from this {domain}
document — it asks for information that is clearly NOT present in the text
(e.g. demographics, dates, names, values, or fields that the document does
not mention at all).

Document:
{text[:1800]}

Output exactly:
Q: <unanswerable question>"""


CANNED_REFUSALS = (
    "Not specified in the document.",
    "The document does not provide this information.",
    "This is not stated in the source material.",
)


def task_refuse(
    teacher_model, teacher_tok, items: list[dict[str, Any]], domain: str,
    *, batch_size: int = 8, max_examples: int = 150,
) -> list[dict[str, str]]:
    """Teacher-generated refusals. Trains the model to confidently reject
    questions whose answers aren't in the source — reduces hallucination."""
    sample = items[:max_examples]
    if not sample:
        return []
    prompts = [make_refuse_prompt(domain, it["original"]) for it in sample]

    rows: list[dict[str, str]] = []
    for start in range(0, len(prompts), batch_size):
        batch_items = sample[start:start + batch_size]
        batch_prompts = prompts[start:start + batch_size]
        outputs = batch_ask(teacher_model, teacher_tok, batch_prompts, max_tokens=80)
        for i, (item, out) in enumerate(zip(batch_items, outputs)):
            q_text = out.replace("Q:", "").strip().split("\n")[0].strip()
            if not q_text or len(q_text) < 10:
                continue
            answer = CANNED_REFUSALS[(start + i) % len(CANNED_REFUSALS)]
            rows.append({
                "task": "refuse",
                "instruction": q_text[:200],
                "input": item["original"][:2500],
                "output": answer,
            })
    return rows


# ── Balancer + orchestrator ──────────────────────────────────────────────


# Balancing was needed for medical (summary+red_flags drowned out rarer
# tasks). For narrow extraction domains (logs, structured data), balancing
# HURTS — it dilutes the dominant task (extract), which is the skill we
# actually need. So we cap only at extreme dominance (>80% of total) and
# only kick in when n_tasks > 1.
MAX_PER_TASK_FRAC = 0.8  # only cap if a task is >80% of total volume


def _balance_curriculum(
    rows: list[dict[str, str]], n_docs: int
) -> list[dict[str, str]]:
    """Cap any task that exceeds MAX_PER_TASK_FRAC of total examples.

    Conservative — only kicks in when one task overwhelmingly dominates.
    Otherwise leaves the natural distribution alone, which preserves the
    dominant skill for narrow domains.
    """
    if not rows:
        return rows
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    if len(by_task) <= 1:
        return rows  # nothing to balance against

    total = len(rows)
    cap = int(total * MAX_PER_TASK_FRAC)

    balanced: list[dict[str, str]] = []
    for task, task_rows in by_task.items():
        if len(task_rows) > cap:
            sampled = random.sample(task_rows, cap)
            print(f"  balancing: {task} {len(task_rows)} → {cap} (>80% dominance)", flush=True)
            balanced.extend(sampled)
        else:
            balanced.extend(task_rows)
    return balanced


def generate_curriculum(
    teacher_model,
    teacher_tok,
    structured_items: list[dict[str, Any]],
    questions: list[str],
    domain: str,
    *,
    on_progress: Callable[[str, int, int], None] | None = None,
    include_new_tasks: bool = True,  # kept for back-compat with run.py CLI flag
) -> list[dict[str, str]]:
    """Run the 6 universal tasks and return the combined training set.

    `include_new_tasks` is preserved for run.py CLI compatibility but no longer
    gates anything — every task here is universal and runs on every domain.
    """

    def _emit(task_name: str, rows: list[dict[str, str]]) -> None:
        print(f"  [{task_name}] generated {len(rows)} examples", flush=True)
        if on_progress is not None:
            on_progress(task_name, len(rows), len(structured_items))

    all_rows: list[dict[str, str]] = []

    # 1. EXTRACT — self-generating, no teacher calls
    rows = task_extract(structured_items, domain)
    _emit("1.extract", rows)
    all_rows.extend(rows)

    # 2-6. Teacher-generated tasks
    rows = task_summarize(teacher_model, teacher_tok, structured_items, domain)
    _emit("2.summarize", rows)
    all_rows.extend(rows)

    rows = task_qa(teacher_model, teacher_tok, structured_items, questions, domain)
    _emit("3.qa", rows)
    all_rows.extend(rows)

    rows = task_paraphrase(teacher_model, teacher_tok, structured_items, domain)
    _emit("4.paraphrase", rows)
    all_rows.extend(rows)

    rows = task_yes_no(teacher_model, teacher_tok, structured_items, domain)
    _emit("5.yes_no", rows)
    all_rows.extend(rows)

    rows = task_refuse(teacher_model, teacher_tok, structured_items, domain)
    _emit("6.refuse", rows)
    all_rows.extend(rows)

    print(f"\n✅ Curriculum raw: {len(all_rows)} examples across 6 tasks", flush=True)

    all_rows = _balance_curriculum(all_rows, n_docs=len(structured_items))
    print(f"✅ Curriculum balanced: {len(all_rows)} training examples", flush=True)
    return all_rows
