"""Q&A generation (step5_fix.py) + filter (filter_data.py).

Generates real LLM answers per (doc, question) pair, then drops "Not specified"
junk. Without this filter, training tanks (validated empirically — 26.9% vs 64.7%).
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .llm import batch_ask


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

    Returns rows shaped like {instruction, input, output} ready for SFT.
    """
    pairs: list[dict[str, Any]] = []
    for item in structured:
        for q in questions:
            pairs.append(
                {
                    "original": item["original"],
                    "structured": item["structured"],
                    "question": q,
                }
            )

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


def filter_clean(training_data: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop "Not specified" / unknown / refusal / too-short rows.

    Same rules as filter_data.py. CRITICAL — without this, the model trains on
    junk answers and inference quality collapses.
    """
    filtered: list[dict[str, str]] = []
    for d in training_data:
        out = d["output"].lower().strip()
        if "not specified" in out:
            continue
        if "unknown" in out[:30]:
            continue
        if len(d["output"]) < 50:
            continue
        if out.startswith(("i ", "i'm", "i can", "sorry")):
            continue
        filtered.append(d)
    return filtered
