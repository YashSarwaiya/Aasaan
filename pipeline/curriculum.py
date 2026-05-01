"""Multi-task training curriculum.

Replaces the single Q&A generation step with 8 different training tasks. Each
task teaches a different clinical skill, so the resulting LoRA learns to
*reason* about notes rather than just answer the same 10 question phrasings.

Tasks:
  1. STRUCTURE       — raw note → structured form (JSON)
  2. CONTINUATION    — note up to "PLAN:" → the plan
  3. ASSESSMENT      — note minus assessment → assessment text
  4. DIFFERENTIAL    — symptoms + history → differential diagnosis
  5. RED_FLAGS       — full note → critical concerns
  6. PATTERN         — N similar notes → common pattern
  7. SUMMARY         — clinical note → plain-English summary
  8. QA              — structured form + question → direct answer

Tasks 1-3 are self-generating (text manipulation on existing extracted data,
no LLM calls). Tasks 4-8 use the teacher (Qwen 32B) to produce the output.

Each row is shaped like {instruction, input, output, task} ready for SFT.
The `task` field is for analysis only — `train.py` ignores it.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from typing import Any, Callable

from .llm import batch_ask
from .qa import EMPTY_VALUES, has_content_for_question, is_empty_value


# ── Helpers ──────────────────────────────────────────────────────────────


# Section headers we look for to split a clinical note into pieces.
# Order matters — we try longer/more-specific markers before generic ones.
PLAN_MARKERS = (
    "PLAN:", "Plan:", "TREATMENT PLAN:", "Treatment Plan:",
    "Treatment plan:", "RECOMMENDATIONS:", "Recommendations:",
)
ASSESSMENT_MARKERS = (
    "ASSESSMENT:", "Assessment:",
    "IMPRESSION:", "Impression:",
    "ASSESSMENT AND PLAN:", "Assessment and Plan:",
)


def _split_on_marker(note: str, markers: tuple[str, ...]) -> tuple[str, str] | None:
    """Split note at the first matching marker. Returns (before, after) or None."""
    for marker in markers:
        idx = note.find(marker)
        if idx >= 0:
            before = note[:idx].rstrip()
            after = note[idx + len(marker):].lstrip()
            if len(before) > 100 and len(after) > 30:
                return before, after
    return None


def _structured_subset(structured: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """Return only the fields in `fields` that have non-empty content."""
    return {
        f: structured[f]
        for f in fields
        if f in structured and not is_empty_value(structured[f])
    }


# ── Task 1: STRUCTURE — raw note → structured form ───────────────────────


def task_structure(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Self-generating: input = raw note, output = structured form (already extracted)."""
    rows: list[dict[str, str]] = []
    for item in items:
        if is_empty_value(item.get("structured")):
            continue
        rows.append({
            "task": "structure",
            "instruction": "Extract the structured clinical form from this note. Output JSON.",
            "input": item["original"][:2500],
            "output": json.dumps(item["structured"], indent=2)[:1500],
        })
    return rows


# ── Task 2: CONTINUATION — partial note → plan ───────────────────────────


def task_continuation(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Self-generating: split at PLAN: marker, train model to write the plan."""
    rows: list[dict[str, str]] = []
    for item in items:
        split = _split_on_marker(item["original"], PLAN_MARKERS)
        if split is None:
            continue
        before, plan = split
        rows.append({
            "task": "continuation",
            "instruction": "Continue this clinical note. Write the PLAN section.",
            "input": before[-2000:],
            "output": plan[:600],
        })
    return rows


# ── Task 3: ASSESSMENT — note minus assessment → assessment ──────────────


def task_assessment_extraction(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Self-generating: extract the ASSESSMENT/IMPRESSION section as ground truth."""
    rows: list[dict[str, str]] = []
    for item in items:
        split = _split_on_marker(item["original"], ASSESSMENT_MARKERS)
        if split is None:
            continue
        before, assessment_block = split
        # Trim assessment to first paragraph or 600 chars (whichever is shorter)
        assessment = assessment_block.split("\n\n")[0][:600]
        if len(assessment) < 30:
            continue
        rows.append({
            "task": "assessment",
            "instruction": "Write the clinical ASSESSMENT for this case based on the history and exam.",
            "input": before[-2000:],
            "output": assessment,
        })
    return rows


# ── Task 4: DIFFERENTIAL — symptoms → differential diagnosis ─────────────


def make_differential_prompt(structured: dict[str, Any]) -> str:
    """Teacher prompt: given symptoms + history, generate a differential."""
    relevant_keys = (
        "chief_complaint", "history_of_present_illness",
        "past_medical_history", "physical_examination",
        "vitals", "laboratory_data",
    )
    context = _structured_subset(structured, list(relevant_keys))
    return f"""You are a senior clinician.

Patient information:
{json.dumps(context, indent=2)[:1500]}

List 3-5 possible diagnoses for this patient, ordered by likelihood. For each, give a 1-sentence reasoning.

Output as a numbered list. Do NOT include a final answer or definitive diagnosis — this is the differential.

Differential diagnosis:"""


def task_differential(
    teacher_model, teacher_tok, items: list[dict[str, Any]],
    *, batch_size: int = 8,
) -> list[dict[str, str]]:
    """Teacher-generated: input = symptoms+history, output = differential list."""
    pairs: list[tuple[dict[str, Any], str]] = []
    for item in items:
        s = item.get("structured", {})
        if is_empty_value(s.get("chief_complaint")) and is_empty_value(s.get("history_of_present_illness")):
            continue
        pairs.append((item, make_differential_prompt(s)))

    rows: list[dict[str, str]] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        prompts = [p[1] for p in batch]
        outputs = batch_ask(teacher_model, teacher_tok, prompts, max_tokens=350)
        for (item, _), out in zip(batch, outputs):
            if len(out.strip()) < 30:
                continue
            rows.append({
                "task": "differential",
                "instruction": "List the most likely differential diagnoses for this patient with brief reasoning.",
                "input": item["original"][:2500],
                "output": out.strip()[:800],
            })
    return rows


# ── Task 5: RED_FLAGS — full note → critical concerns ────────────────────


def make_red_flags_prompt(note: str) -> str:
    return f"""You are a senior clinician reviewing this case for safety concerns.

CLINICAL NOTE:
{note[:2500]}

Identify ANY red flags — symptoms, lab values, history items, or vital signs that warrant urgent attention or escalation.

If there are no red flags, say so explicitly.

Output as a short bullet list (3-6 items). Be specific and grounded in the note — do NOT invent concerns that aren't supported.

Red flags:"""


def task_red_flags(
    teacher_model, teacher_tok, items: list[dict[str, Any]],
    *, batch_size: int = 8,
) -> list[dict[str, str]]:
    """Teacher-generated: input = full note, output = red flags list."""
    pairs: list[tuple[dict[str, Any], str]] = []
    for item in items:
        if len(item.get("original", "")) < 200:
            continue
        pairs.append((item, make_red_flags_prompt(item["original"])))

    rows: list[dict[str, str]] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        prompts = [p[1] for p in batch]
        outputs = batch_ask(teacher_model, teacher_tok, prompts, max_tokens=300)
        for (item, _), out in zip(batch, outputs):
            if len(out.strip()) < 30:
                continue
            rows.append({
                "task": "red_flags",
                "instruction": "Identify red flags or critical concerns in this clinical case.",
                "input": item["original"][:2500],
                "output": out.strip()[:600],
            })
    return rows


# ── Task 6: PATTERN — N similar notes → common pattern ───────────────────


def _cluster_by_complaint(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group notes by primary complaint keywords. Simple but effective."""
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        s = item.get("structured", {})
        cc = s.get("chief_complaint") or s.get("history_of_present_illness") or ""
        if isinstance(cc, list):
            cc = " ".join(str(x) for x in cc)
        if not isinstance(cc, str) or len(cc) < 10:
            continue
        # First 3 meaningful words as cluster key
        words = re.findall(r"\b[a-z]{4,}\b", cc.lower())
        if len(words) < 2:
            continue
        key = " ".join(words[:3])
        clusters[key].append(item)
    # Keep only clusters with 3+ similar cases
    return {k: v for k, v in clusters.items() if len(v) >= 3}


def make_pattern_prompt(notes: list[str]) -> str:
    blob = "\n\n=== CASE SEPARATOR ===\n\n".join(n[:1200] for n in notes[:5])
    return f"""You are reviewing similar cases from one practice.

CASES (notes from multiple patients with similar presenting complaints):
{blob}

Identify the COMMON PATTERN across these cases:
- Typical presenting symptoms
- Common workup or tests ordered
- Usual treatment approach
- Typical disposition / follow-up

Be specific to what these cases actually show. 3-5 bullet points. Do not generalize beyond the cases.

Common pattern:"""


def task_pattern(
    teacher_model, teacher_tok, items: list[dict[str, Any]],
    *, max_clusters: int = 200, batch_size: int = 4,
) -> list[dict[str, str]]:
    """Teacher-generated: cluster similar notes, ask teacher for the common pattern.

    Lower batch_size because input has 5 notes × ~1200 chars = larger context.
    """
    clusters = _cluster_by_complaint(items)
    cluster_list = list(clusters.values())[:max_clusters]
    if not cluster_list:
        return []

    pairs: list[tuple[list[dict[str, Any]], str]] = []
    for cluster in cluster_list:
        sample = random.sample(cluster, min(5, len(cluster)))
        notes = [c["original"] for c in sample]
        pairs.append((sample, make_pattern_prompt(notes)))

    rows: list[dict[str, str]] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        prompts = [p[1] for p in batch]
        outputs = batch_ask(teacher_model, teacher_tok, prompts, max_tokens=400)
        for (sample, _), out in zip(batch, outputs):
            if len(out.strip()) < 50:
                continue
            # Use the FIRST note as the input — model learns to recognize the
            # pattern from any one note that fits the cluster.
            rows.append({
                "task": "pattern",
                "instruction": "What is the common clinical pattern for cases like this in this practice?",
                "input": sample[0]["original"][:2500],
                "output": out.strip()[:700],
            })
    return rows


# ── Task 7: SUMMARY — clinical note → plain-English summary ──────────────


def make_summary_prompt(note: str) -> str:
    return f"""Translate this clinical note into plain English a patient could understand.

CLINICAL NOTE:
{note[:2500]}

Write 3-5 sentences. No medical jargon — explain conditions, medications, and next steps in everyday language. Keep all key facts intact.

Plain English summary:"""


def task_summary(
    teacher_model, teacher_tok, items: list[dict[str, Any]],
    *, batch_size: int = 8,
) -> list[dict[str, str]]:
    """Teacher-generated: input = clinical note, output = plain English."""
    pairs: list[tuple[dict[str, Any], str]] = []
    for item in items:
        if len(item.get("original", "")) < 200:
            continue
        pairs.append((item, make_summary_prompt(item["original"])))

    rows: list[dict[str, str]] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        prompts = [p[1] for p in batch]
        outputs = batch_ask(teacher_model, teacher_tok, prompts, max_tokens=300)
        for (item, _), out in zip(batch, outputs):
            if len(out.strip()) < 50:
                continue
            rows.append({
                "task": "summary",
                "instruction": "Summarize this clinical case in plain English a patient could understand.",
                "input": item["original"][:2500],
                "output": out.strip()[:600],
            })
    return rows


# ── Task 8: Q&A — structured form + question → answer ────────────────────
# Uses the same skip-empty + has_content_for_question helpers as the legacy
# qa.py path, but only over REASONING-classified questions so we don't burn
# training capacity on lookup answers (those go to RAG at serving time).


def make_qa_prompt(domain: str, structured: dict[str, Any], question: str) -> str:
    context = json.dumps(structured, indent=2)[:1500]
    return f"""Domain: {domain}

Structured data:
{context}

Question: {question}

Give a concise direct answer (1-3 sentences). Only state facts grounded in the structured data. If info is missing, say "Not specified in document."
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
    """Teacher-generated: traditional Q&A pairs, sampled over reasoning questions only."""
    if not questions:
        return []

    pairs: list[tuple[dict[str, Any], str, str]] = []
    for item in items:
        # Sample N reasoning questions per doc that actually have content.
        candidate = [q for q in questions if has_content_for_question(item["structured"], q)]
        if not candidate:
            continue
        sample = candidate[:questions_per_doc]
        for q in sample:
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


# ── Top-level orchestrator ───────────────────────────────────────────────


# Cap each task at this fraction of total docs to prevent summary/red_flags
# from drowning out the rarer-but-important reasoning tasks (assessment,
# continuation). Empirically determined: v3 had 63% summary+red_flags which
# made the trained model behave like a summary writer instead of a clinician.
MAX_PER_TASK_FRAC = 0.5  # cap any single task at 50% of #docs


def _balance_curriculum(
    rows: list[dict[str, str]], n_docs: int
) -> list[dict[str, str]]:
    """Cap any task that exceeds MAX_PER_TASK_FRAC * n_docs.

    Random-sample down to the cap so the model gets a balanced curriculum.
    Tasks below the cap are left untouched.
    """
    cap = max(int(n_docs * MAX_PER_TASK_FRAC), 50)
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    balanced: list[dict[str, str]] = []
    for task, task_rows in by_task.items():
        if len(task_rows) > cap:
            sampled = random.sample(task_rows, cap)
            print(f"  balancing: {task} {len(task_rows)} → {cap}")
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
) -> list[dict[str, str]]:
    """Run all 8 tasks and return the combined training set.

    Self-generating tasks (1-3) run first — free, fast. Then teacher tasks
    (4-8) run sequentially against the loaded teacher model.
    """

    def _emit(task_name: str, rows: list[dict[str, str]]) -> None:
        print(f"  [{task_name}] generated {len(rows)} examples", flush=True)
        if on_progress is not None:
            on_progress(task_name, len(rows), len(structured_items))

    all_rows: list[dict[str, str]] = []

    # Tasks 1-3: self-generating, no LLM calls
    rows = task_structure(structured_items)
    _emit("1.structure", rows)
    all_rows.extend(rows)

    rows = task_continuation(structured_items)
    _emit("2.continuation", rows)
    all_rows.extend(rows)

    rows = task_assessment_extraction(structured_items)
    _emit("3.assessment", rows)
    all_rows.extend(rows)

    # Tasks 4-8: teacher (Qwen 32B) calls
    rows = task_differential(teacher_model, teacher_tok, structured_items)
    _emit("4.differential", rows)
    all_rows.extend(rows)

    rows = task_red_flags(teacher_model, teacher_tok, structured_items)
    _emit("5.red_flags", rows)
    all_rows.extend(rows)

    rows = task_pattern(teacher_model, teacher_tok, structured_items)
    _emit("6.pattern", rows)
    all_rows.extend(rows)

    rows = task_summary(teacher_model, teacher_tok, structured_items)
    _emit("7.summary", rows)
    all_rows.extend(rows)

    rows = task_qa(teacher_model, teacher_tok, structured_items, questions, domain)
    _emit("8.qa", rows)
    all_rows.extend(rows)

    print(f"\n✅ Curriculum raw: {len(all_rows)} examples across 8 tasks", flush=True)

    # Balance — prevent any one task from dominating training
    all_rows = _balance_curriculum(all_rows, n_docs=len(structured_items))
    print(f"✅ Curriculum balanced: {len(all_rows)} training examples", flush=True)
    return all_rows
