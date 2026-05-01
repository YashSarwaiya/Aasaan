"""Q&A generation (step5_fix.py) + filter (filter_data.py).

Generates real LLM answers per (doc, question) pair, then drops "Not specified"
junk. Without this filter, training tanks (validated empirically — 26.9% vs 64.7%).

Two layers of defense:
  1. Upfront: skip (doc, question) pairs where structured form has no relevant
     content. Saves LLM compute and prevents junk Q&A from entering pipeline.
  2. Post-filter: drop generated answers that contain refusal/junk patterns.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .llm import batch_ask


# Values that mean "this field is empty / unknown / not in document".
# IMPORTANT: do NOT include "none" — "None" is a real answer ("patient is on no
# medications") and is informative for training.
EMPTY_VALUES = {
    "", "unknown", "not specified", "not mentioned", "not provided",
    "n/a", "na", "not applicable", "not available", "none mentioned",
}


def is_empty_value(value: Any) -> bool:
    """True if the field is effectively empty (no info to answer from)."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in EMPTY_VALUES
    if isinstance(value, list):
        return len(value) == 0 or all(is_empty_value(v) for v in value)
    if isinstance(value, dict):
        return len(value) == 0 or all(is_empty_value(v) for v in value.values())
    return False


# Keyword groups: any question containing a synonym maps to fields containing the key.
# Lets us skip "What medications?" if the only medication-ish field is empty.
FIELD_KEYWORD_MAP = {
    "diagnos":   ["diagnos", "condition", "wrong with", "what does the patient have"],
    "medic":     ["medic", "drug", "rx", "prescription", "pharma", "what is the patient on", "taking"],
    "plan":      ["plan", "treatment", "next step", "follow up", "follow-up", "recommend", "what should"],
    "age":       ["age", "old", "years old"],
    "name":      ["name", "who is the patient"],
    "history":   ["history", "past", "background", "presenting"],
    "vital":     ["vital", "blood pressure", "heart rate", "bp", "hr", "temperature", "pulse"],
    "exam":      ["exam", "physical", "findings on examination"],
    "lab":       ["lab", "laboratory", "test result", "blood work", "blood test"],
    "procedure": ["procedure", "operation", "surgery", "intervention", "performed"],
    "indication":["indication", "why was", "reason for"],
    "complic":   ["complic", "risk", "adverse", "side effect"],
    "impression":["impression", "assessment", "summary"],
}


def has_content_for_question(structured: dict[str, Any], question: str) -> bool:
    """Skip Q&A generation if no relevant field has content.

    Heuristic — find which schema fields the question is asking about, then
    check if any of those fields actually has non-empty content. Conservative:
    if the question doesn't match any known keyword group, allow it through
    (better to generate a borderline pair than skip a valid one).
    """
    q_lower = question.lower()

    # Which keyword groups does the question hit?
    matched_groups = [
        group_key for group_key, synonyms in FIELD_KEYWORD_MAP.items()
        if any(syn in q_lower for syn in synonyms)
    ]
    if not matched_groups:
        # Generic question with no clear field mapping — let the LLM try.
        return True

    # Find schema fields whose name contains one of the matched group keys.
    relevant_fields = [
        f for f in structured.keys()
        if any(group_key in f.lower() for group_key in matched_groups)
    ]
    if not relevant_fields:
        # Schema has no field for this question topic — let the LLM try the
        # full structured form anyway (it might pull from additional_notes etc.).
        return True

    # Skip only if EVERY relevant field is empty.
    return any(not is_empty_value(structured[f]) for f in relevant_fields)


def make_qa_prompt(domain: str, structured: dict[str, Any], question: str) -> str:
    context = json.dumps(structured, indent=2)[:1500]
    return f"""Domain: {domain}

Structured data:
{context}

Question: {question}

Give a concise direct answer (1-3 sentences). If info is missing, say "Not specified in document."
Answer:"""


def generate_qa(
    model,
    tokenizer,
    structured: list[dict[str, Any]],
    questions: list[str],
    domain: str,
    *,
    batch_size: int = 16,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, str]]:
    """For every (doc, question) pair, ask the LLM for a real answer.

    Pairs where the structured form has no relevant content are skipped upfront.
    Returns rows shaped like {instruction, input, output} ready for SFT.
    """
    pairs: list[dict[str, Any]] = []
    skipped = 0
    for item in structured:
        for q in questions:
            if not has_content_for_question(item["structured"], q):
                skipped += 1
                continue
            pairs.append(
                {
                    "original": item["original"],
                    "structured": item["structured"],
                    "question": q,
                }
            )

    if skipped:
        # Print so it shows up in SLURM logs and we can see the impact.
        print(f"  skipped {skipped} (doc, question) pairs — relevant field empty")

    training_data: list[dict[str, str]] = []
    for batch_start in range(0, len(pairs), batch_size):
        batch = pairs[batch_start : batch_start + batch_size]
        prompts = [make_qa_prompt(domain, p["structured"], p["question"]) for p in batch]
        answers = batch_ask(model, tokenizer, prompts, max_tokens=150)

        for p, ans in zip(batch, answers):
            training_data.append(
                {
                    "instruction": p["question"],
                    "input": p["original"][:2500],
                    "output": ans,
                }
            )

        if on_progress is not None:
            on_progress(min(batch_start + batch_size, len(pairs)), len(pairs))

    return training_data


# Phrases that indicate a refusal / "I don't know" answer rather than real info.
# Conservative — only patterns that almost always mean the answer is junk.
JUNK_PATTERNS = [
    "not specified",
    "not mentioned in the",
    "cannot be determined",
    "the document does not",
    "the note does not",
    "the report does not",
    "no information about",
    "no information regarding",
    "is not provided in",
]

# Placeholder strings the LLM regurgitates from anonymized source data.
PLACEHOLDER_PATTERNS = [
    "mm/dd/yyyy", "xx/xx", "[name]", "<date>", "[redacted]", "abcd1234",
]


def filter_clean(training_data: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop refusal / placeholder / too-short rows.

    Two-layer defense alongside has_content_for_question(). CRITICAL — without
    this, the model trains on junk answers and inference quality collapses.
    """
    filtered: list[dict[str, str]] = []
    for d in training_data:
        out_lower = d["output"].lower().strip()

        # Refusals / "I don't know" answers
        if any(p in out_lower for p in JUNK_PATTERNS):
            continue
        # Anonymization placeholders leaking through (case-insensitive)
        if any(p in out_lower for p in PLACEHOLDER_PATTERNS):
            continue
        # "Unknown" near the start = refusal
        if "unknown" in out_lower[:50]:
            continue
        # Too short to be a useful training example
        if len(d["output"]) < 50:
            continue
        # First-person assistant deflection
        if out_lower.startswith(("i ", "i'm", "i can", "sorry", "as an ai")):
            continue

        filtered.append(d)
    return filtered


# ── Grounding validator (Qwen 32B as judge) ──────────────────────────────


GROUNDING_PROMPT = """You are a strict but fair clinical fact checker.

Given the source clinical note and an answer the AI produced, score whether the answer is GROUNDED in the note.

=== RUBRIC ===
1 = every fact in the answer is in the note (or correctly states the info is missing)
0 = the answer contains made-up facts, hallucinated drugs/dates/conditions, or contradicts the note

A confident "no medications" answer when the note says "PERTINENT MEDICATION: None" is GROUNDED (1).
An answer that mentions a drug name not in the note is HALLUCINATED (0).
An answer with placeholder text like "mm/dd/yyyy" is HALLUCINATED (0).
An answer that adds clinical reasoning consistent with the note's facts is GROUNDED (1).
An answer that invents specific numbers, dates, or names not in the note is HALLUCINATED (0).

=== CASE ===
NOTE:
{note}

INSTRUCTION: {instruction}
ANSWER: {answer}

Apply the rubric. Respond with ONLY a single character: 1 or 0
Score:"""


def _parse_score(response: str) -> int:
    """Extract first 0/1 from judge output. Default to 1 (keep) on parse failure
    so we don't drop pairs because of formatting glitches in the judge."""
    import re
    m = re.search(r"[01]", response)
    return int(m.group()) if m else 1


def validate_grounded(
    judge_model,
    judge_tok,
    training_data: list[dict[str, str]],
    *,
    batch_size: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, str]], int]:
    """Validate every training pair against its source note using a teacher LLM.

    Same pattern as `judge.py` (the eval scorer). Returns (kept_pairs, dropped_count).
    The judge model should be Qwen 32B — generates calls expensive but only
    ~4 max-tokens per call so it's fast.

    SAFETY: parse failures default to KEEP (1), so a malformed judge response
    can't accidentally wipe out training data.
    """
    if not training_data:
        return [], 0

    prompts = [
        GROUNDING_PROMPT.format(
            note=d["input"][:2000],
            instruction=d["instruction"],
            answer=d["output"][:600],
        )
        for d in training_data
    ]

    kept: list[dict[str, str]] = []
    dropped = 0
    total = len(training_data)
    for start in range(0, total, batch_size):
        batch_data = training_data[start:start + batch_size]
        batch_prompts = prompts[start:start + batch_size]
        responses = batch_ask(judge_model, judge_tok, batch_prompts, max_tokens=4)
        for d, resp in zip(batch_data, responses):
            score = _parse_score(resp)
            if score == 1:
                kept.append(d)
            else:
                dropped += 1
        if on_progress is not None:
            on_progress(min(start + batch_size, total), total)

    print(
        f"  validated {total} pairs → kept {len(kept)}, dropped {dropped} hallucinated",
        flush=True,
    )
    return kept, dropped
