"""Multi-pass Q&A refinement.

After initial generation, each (question, answer) pair runs through:
  1. CRITIQUE — teacher LLM identifies what's wrong (vague, ungrounded,
     missing detail, wrong tone)
  2. REWRITE — teacher LLM produces an improved answer addressing the
     critique while staying grounded in the source note

This is the single biggest data-quality upgrade. The original answer was
the teacher's first attempt; the refined answer is the teacher's better
attempt after self-criticism. Industry research shows refined answers
yield +5-15pp accuracy when used as training labels.

Idea borrowed from Augmentoolkit's "self-correction" pipeline and Anthropic's
constitutional AI.
"""

from __future__ import annotations

from typing import Any, Callable

from .llm import batch_ask


CRITIQUE_PROMPT = """You are a strict reviewer of clinical Q&A training data.

Source note:
{note}

Question: {question}
Answer: {answer}

Critique this answer in ONE sentence. Focus on:
- Is it grounded in the source note (no invented facts)?
- Is it specific and concrete (not vague)?
- Is it the right length (not too short, not rambling)?
- Does it actually answer the question (not deflect)?

If the answer is already good, say "ANSWER IS GOOD".
Otherwise, name the single biggest flaw.

Critique:"""


REWRITE_PROMPT = """You are improving a clinical Q&A training pair.

Source note:
{note}

Question: {question}
Original answer: {answer}
Critique of original: {critique}

Rewrite the answer to address the critique. Rules:
- Use ONLY facts present in the source note
- Be concise (1-3 sentences)
- Answer the question directly — no hedging, no preamble
- If the note doesn't have the info, say "Not in the note" briefly

Improved answer:"""


def refine_pairs(
    teacher_model,
    teacher_tok,
    pairs: list[dict[str, str]],
    *,
    batch_size: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, str]]:
    """Run critique → rewrite on each Q&A pair. Returns same shape with
    `output` replaced by the rewritten version. Pairs with critique
    "ANSWER IS GOOD" are passed through unchanged.

    Each pair = 2 LLM calls (critique + rewrite). 2x compute for
    significantly higher data quality.
    """
    if not pairs:
        return []

    refined: list[dict[str, str]] = []
    n = len(pairs)

    # Stage 1: critique each pair in batches
    print(f"  refining {n} pairs (stage 1: critique)...", flush=True)
    critiques: list[str] = []
    for start in range(0, n, batch_size):
        batch = pairs[start:start + batch_size]
        prompts = [
            CRITIQUE_PROMPT.format(
                note=p["input"][:1500],
                question=p["instruction"],
                answer=p["output"][:600],
            )
            for p in batch
        ]
        responses = batch_ask(teacher_model, teacher_tok, prompts, max_tokens=80)
        critiques.extend(responses)
        if on_progress is not None:
            on_progress(min(start + batch_size, n), n * 2)

    # Stage 2: rewrite pairs that had critiques (skip "ANSWER IS GOOD")
    print(f"  refining {n} pairs (stage 2: rewrite)...", flush=True)
    needs_rewrite_idx = [
        i for i, c in enumerate(critiques)
        if "ANSWER IS GOOD" not in c.upper()
    ]
    print(f"  {len(needs_rewrite_idx)}/{n} pairs need rewriting "
          f"({n - len(needs_rewrite_idx)} were already good)", flush=True)

    rewrites: dict[int, str] = {}
    for batch_start in range(0, len(needs_rewrite_idx), batch_size):
        batch_idxs = needs_rewrite_idx[batch_start:batch_start + batch_size]
        prompts = [
            REWRITE_PROMPT.format(
                note=pairs[i]["input"][:1500],
                question=pairs[i]["instruction"],
                answer=pairs[i]["output"][:600],
                critique=critiques[i][:300],
            )
            for i in batch_idxs
        ]
        responses = batch_ask(teacher_model, teacher_tok, prompts, max_tokens=200)
        for idx, resp in zip(batch_idxs, responses):
            rewrites[idx] = resp.strip()
        if on_progress is not None:
            on_progress(n + min(batch_start + batch_size, len(needs_rewrite_idx)), n * 2)

    # Combine: keep original answer if good, swap in rewrite otherwise
    for i, p in enumerate(pairs):
        new = dict(p)
        if i in rewrites and len(rewrites[i]) > 20:
            new["output"] = rewrites[i]
            new["refined"] = True
        else:
            new["refined"] = False
        refined.append(new)

    n_refined = sum(1 for p in refined if p.get("refined"))
    print(f"  ✅ refinement complete: {n_refined}/{n} pairs improved", flush=True)
    return refined
