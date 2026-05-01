"""Quality classifier for training pairs.

Beyond the binary grounded/hallucinated check (qa.validate_grounded), this
module asks the teacher LLM for a more nuanced 0-5 quality score covering:
  - Groundedness in source
  - Specificity (concrete facts vs hand-waving)
  - Completeness (answers the question fully)
  - Conciseness (no rambling)
  - Tone (clinical / on-style)

Drops pairs below a threshold (default 3/5). Result: training data is
"good answers", not just "non-hallucinated answers".

Inspired by NeMo Curator's quality classifier and the FineWeb-Edu paper's
educational-quality model approach.
"""

from __future__ import annotations

import re
from typing import Callable

from .llm import batch_ask


QUALITY_PROMPT = """You are a strict reviewer of clinical Q&A training data.

Source note:
{note}

Question: {question}
Answer: {answer}

Score this Q&A pair from 0-5 based on training quality:
  5 = Excellent (grounded, specific, concise, well-toned, fully answers the question)
  4 = Good (minor issues but training-worthy)
  3 = Acceptable (usable but mediocre)
  2 = Weak (significant flaws — vague, partial, slightly off)
  1 = Bad (mostly wrong, missing, or off-topic)
  0 = Garbage (hallucinated, contradicts note, or refuses)

Output ONLY the integer score (0-5). Nothing else.
Score:"""


def _parse_score(response: str) -> int:
    """Extract first 0-5 from response. Default to 3 (keep) on parse failure."""
    m = re.search(r"\b([0-5])\b", response)
    return int(m.group(1)) if m else 3


def score_quality(
    judge_model,
    judge_tok,
    pairs: list[dict[str, str]],
    *,
    batch_size: int = 16,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[tuple[dict[str, str], int]]:
    """Score every pair 0-5. Returns [(pair, score)]. Doesn't filter.

    Use filter_by_quality() to drop low-scorers.
    """
    if not pairs:
        return []

    scored: list[tuple[dict[str, str], int]] = []
    n = len(pairs)
    for start in range(0, n, batch_size):
        batch = pairs[start:start + batch_size]
        prompts = [
            QUALITY_PROMPT.format(
                note=p["input"][:1500],
                question=p["instruction"],
                answer=p["output"][:600],
            )
            for p in batch
        ]
        responses = batch_ask(judge_model, judge_tok, prompts, max_tokens=4)
        for p, resp in zip(batch, responses):
            scored.append((p, _parse_score(resp)))
        if on_progress is not None:
            on_progress(min(start + batch_size, n), n)

    return scored


def filter_by_quality(
    scored: list[tuple[dict[str, str], int]],
    *,
    min_score: int = 3,
) -> tuple[list[dict[str, str]], dict[int, int]]:
    """Drop pairs scoring below `min_score`. Returns (kept, score_distribution).

    Default min_score=3 keeps "Acceptable and above". More aggressive
    filters (4+) yield smaller but cleaner datasets.
    """
    kept = [p for p, score in scored if score >= min_score]
    distribution: dict[int, int] = {}
    for _, score in scored:
        distribution[score] = distribution.get(score, 0) + 1

    print(
        f"  quality filter (min={min_score}): kept {len(kept)}/{len(scored)} pairs",
        flush=True,
    )
    print(f"  score distribution: {dict(sorted(distribution.items()))}", flush=True)
    return kept, distribution
