"""Batched structured extraction. Ports step 4 of universal_pipeline.py."""

from __future__ import annotations

import json
from typing import Any, Callable

from .llm import batch_ask, extract_json


def make_extract_prompt(domain: str, schema_str: str, doc: str) -> str:
    return f"""Domain: {domain}

Extract info from this document into the form.

FORM FIELDS:
{schema_str}

DOCUMENT:
{doc[:2500]}

Output ONLY valid JSON with fields filled. Use "unknown" if info missing.
JSON:"""


def batch_extract_structured(
    model,
    tokenizer,
    docs: list[str],
    domain: str,
    schema: dict[str, str],
    *,
    batch_size: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Extract structured records from raw docs.

    Returns (results, failed_count). Drops rows whose JSON parse failed or had
    fewer than 3 filled fields — same heuristic as universal_pipeline.py.
    """
    schema_str = json.dumps(schema, indent=2)
    results: list[dict[str, Any]] = []
    failed = 0

    for batch_start in range(0, len(docs), batch_size):
        batch = docs[batch_start : batch_start + batch_size]
        prompts = [make_extract_prompt(domain, schema_str, d) for d in batch]
        responses = batch_ask(model, tokenizer, prompts, max_tokens=800)

        for i, resp in enumerate(responses):
            parsed = extract_json(resp)
            if parsed and len(parsed) >= 3:
                results.append(
                    {
                        "id": batch_start + i,
                        "original": batch[i],
                        "structured": parsed,
                    }
                )
            else:
                failed += 1

        if on_progress is not None:
            on_progress(min(batch_start + batch_size, len(docs)), len(docs))

    return results, failed
