"""LLM helpers — same shape as the HiPerGator scripts.

`ask` and `batch_ask` mirror universal_pipeline.py and step5_fix.py 1:1, so
behavior on a deployed Modal H100 should match what was validated locally.
"""

from __future__ import annotations

import json
from typing import Any

import torch


def load_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Load Qwen base model + tokenizer with bf16 + auto device map."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def ask(model, tokenizer, prompt: str, max_tokens: int = 500) -> str:
    """Single-prompt LLM call. Mirrors universal_pipeline.py:ask()."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=6000
    ).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,
        )
    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


def batch_ask(
    model, tokenizer, prompts: list[str], max_tokens: int = 400
) -> list[str]:
    """Batched LLM call. Mirrors universal_pipeline.py:batch_ask().

    8x faster than serial calls for typical batch sizes.
    """
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=4096
    ).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,
        )
    results = []
    for out in outputs:
        gen = out[inputs.input_ids.shape[1]:]
        results.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
    return results


def extract_json(text: str) -> dict[str, Any] | None:
    """Robust JSON object extraction. Returns None on failure."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        candidate = text[start:end]
        return json.loads(candidate)
    except Exception:
        try:
            cleaned = (
                candidate.replace("```json", "").replace("```", "").strip()  # type: ignore[name-defined]
            )
            return json.loads(cleaned)
        except Exception:
            return None


def extract_json_list(text: str) -> list[Any] | None:
    """Robust JSON list extraction. Returns None on failure."""
    try:
        start = text.find("[")
        end = text.rfind("]") + 1
        return json.loads(text[start:end])
    except Exception:
        return None
