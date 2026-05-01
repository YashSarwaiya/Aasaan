"""DPO (Direct Preference Optimization) data generation.

After SFT (standard fine-tuning), DPO trains the model on `(question,
preferred_answer, rejected_answer)` triples. The model learns to PREFER
the preferred over the rejected. This is what RLHF tries to do without
needing a reward model.

How we generate the pairs:
  - The "preferred" answer is our refined teacher answer (from refine.py
    or the teacher's clean output)
  - The "rejected" answer is constructed by prompting a smaller / less
    constrained model to give a typical-but-flawed answer:
      * vague hedge ("It depends...")
      * hallucinated specifics ("Patient is on aspirin")
      * wrong-on-purpose answer
      * stock-baseline-style verbose rambling

Industry: DPO commonly adds +5-10pp on instruction-following benchmarks
beyond SFT alone. Our specific use case (medical Q&A with hallucination
issues) should benefit even more — DPO directly trains the model to NOT
hallucinate.

Output shape (compatible with LLaMA-Factory DPO + TRL DPOTrainer):
    {
      "instruction": "...",
      "input": "...",
      "chosen": "preferred answer",
      "rejected": "flawed answer",
    }
"""

from __future__ import annotations

import random
from typing import Any, Callable

from .llm import batch_ask


# Strategy 1: ask the teacher for a deliberately-flawed answer.
# Gives us realistic "wrong" answers that match the kind of mistakes
# a baseline model would make in production.
REJECTED_PROMPT_TEMPLATES = [
    # Vague hedge
    """Source note:
{note}

Question: {question}

Answer this question in a vague, hedging, indirect way. Use phrases like
"it depends," "various factors," "could potentially be." Avoid specifics
even if they're in the note. Do NOT give the actual answer.

Vague answer:""",

    # Verbose rambling (baseline style)
    """Source note:
{note}

Question: {question}

Answer in a long-winded, rambling way. Restate the question. Add
unnecessary preambles like "Based on the information provided" and
"It is important to note that." Make it 4-5 sentences when 1 would do.

Rambling answer:""",

    # Hallucinated specifics
    """Source note:
{note}

Question: {question}

Answer this question by adding plausible-sounding clinical details that
are NOT in the source note. Invent reasonable-sounding drug doses, dates,
or test values. Make it look authoritative.

Hallucinated answer:""",
]


def generate_rejected_answers(
    teacher_model,
    teacher_tok,
    sft_pairs: list[dict[str, str]],
    *,
    batch_size: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, str]]:
    """Convert SFT pairs into DPO triples by generating rejected answers.

    For each SFT pair {instruction, input, output}, randomly pick one of
    3 "bad answer" strategies and ask the teacher for that flawed version.
    Output: {instruction, input, chosen=output, rejected=flawed_answer}.
    """
    if not sft_pairs:
        return []

    n = len(sft_pairs)
    print(f"  generating {n} rejected answers for DPO...", flush=True)

    # Pick a strategy per pair (deterministic via seed for reproducibility)
    rng = random.Random(42)
    strategies = [rng.choice(REJECTED_PROMPT_TEMPLATES) for _ in range(n)]

    dpo_triples: list[dict[str, str]] = []
    for start in range(0, n, batch_size):
        batch = sft_pairs[start:start + batch_size]
        batch_strats = strategies[start:start + batch_size]
        prompts = [
            strat.format(
                note=p["input"][:1500],
                question=p["instruction"],
            )
            for p, strat in zip(batch, batch_strats)
        ]
        rejecteds = batch_ask(teacher_model, teacher_tok, prompts, max_tokens=200)

        for p, rejected in zip(batch, rejecteds):
            rejected = rejected.strip()
            # Sanity: rejected should differ from chosen by enough to be useful
            if len(rejected) < 20 or rejected.lower() == p["output"].lower():
                continue
            dpo_triples.append({
                "instruction": p["instruction"],
                "input": p["input"],
                "chosen": p["output"],
                "rejected": rejected,
            })

        if on_progress is not None:
            on_progress(min(start + batch_size, n), n)

    print(f"  ✅ DPO data: {len(dpo_triples)} preference triples", flush=True)
    return dpo_triples
