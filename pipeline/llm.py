"""LLM helpers — same shape as the HiPerGator scripts.

`ask` and `batch_ask` mirror universal_pipeline.py and step5_fix.py 1:1, so
behavior on a deployed Modal H100 should match what was validated locally.
"""

from __future__ import annotations

import json
from typing import Any

import torch


def load_model(model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """Load student base model + tokenizer with bf16 + auto device map."""
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


def load_teacher(
    model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
    *,
    quantize_4bit: bool | None = None,
):
    """Load the bigger teacher model used by the multi-task curriculum and
    the grounding validator.

    Auto-detects whether 4-bit quantization is needed: if the GPU has < 60 GB
    free, falls back to bnb 4-bit (Llama 70B at 4-bit ≈ 35 GB, fits on a 48 GB
    A6000; tight on 22 GB L4). On B200 (180 GB) it loads in bf16 for full quality.

    `quantize_4bit=True/False` lets the caller force a mode for testing.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if quantize_4bit is None:
        # Auto-decide based on GPU memory
        if torch.cuda.is_available():
            gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            quantize_4bit = gpu_total_gb < 60
        else:
            quantize_4bit = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if quantize_4bit:
        from transformers import BitsAndBytesConfig

        print(f"Loading {model_name} in 4-bit (small GPU detected)", flush=True)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        print(f"Loading {model_name} in bf16 (large GPU)", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    model.eval()
    return model, tokenizer


def unload_model(model) -> None:
    """Free GPU memory before loading a different model.

    Call this between teacher → student swaps in the pipeline. Without it
    the next load_model() can OOM because PyTorch holds onto the previous
    weights' memory until garbage collection runs.
    """
    import gc

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
