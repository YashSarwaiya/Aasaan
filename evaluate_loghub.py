"""LogHub evaluation: base Llama vs each trained LoRA adapter.

Field-level exact-match accuracy against LogHub's structured labels.
NO LLM judge — pure string compare against the ground truth shipped by LogHub.

Usage:
    python evaluate_loghub.py \
        --test-json ./loghub_data/loghub_test.json \
        --adapters output_v3_20260503_*/adapter output_v4_*/adapter \
                   output_v5_*/adapter output_v6_*/adapter \
        --num 200 \
        --output ./eval_loghub_results.json

Output: a comparison table per field (Date/Time/Level/Component/Content/EventTemplate)
plus an overall accuracy + JSON validity %, written both to stdout and to
`eval_loghub_results.json` (summary) + `eval_loghub_results_detail.json` (per-entry).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline.llm import batch_ask, extract_json


BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

def build_parse_prompt(fields: list[str], label_sets: dict[str, list[str]] | None = None) -> str:
    """Build a domain-agnostic extraction prompt from the ground-truth field set.

    Auto-derives which fields to ask for based on what the test set's ground
    truths contain. If `label_sets` is provided (mapping field -> list of
    valid labels), includes them as constraints — turns extraction into
    classification, which lifts accuracy dramatically when the answer space
    is bounded (e.g. CUAD's 41 clause types).

    Adds LogHub-specific guidance (verbatim extraction, `<*>` placeholders)
    only when those fields are present.
    """
    label_sets = label_sets or {}
    field_list = ", ".join(fields)
    rules = [
        "Use the EXACT substring from the input. Do NOT reformat or paraphrase.",
        "If a field is not in the input, omit it from the JSON.",
        "No explanations, no markdown, no preamble. JSON only.",
    ]
    if "EventTemplate" in fields:
        rules.insert(
            1,
            "For EventTemplate, replace variable parts (numbers, IDs, paths, blocks) with <*>.",
        )

    label_constraints = ""
    if label_sets:
        label_blocks = []
        for field, labels in label_sets.items():
            label_blocks.append(
                f"For \"{field}\", pick EXACTLY ONE of these {len(labels)} options "
                f"(case-sensitive, no other values allowed):\n  "
                + "\n  ".join(f"- {l}" for l in labels)
            )
        label_constraints = "\n\nLabel constraints:\n" + "\n\n".join(label_blocks)
        rules.insert(
            0,
            "Use ONLY the exact label values listed in the Label constraints below. Do not invent new labels.",
        )

    rules_str = "\n- ".join([""] + rules).strip()

    return (
        "You are an extraction tool. Parse this input into JSON.\n\n"
        f"Output ONLY valid JSON with these fields if present (omit any not in the input):\n"
        f"- {field_list}{label_constraints}\n\n"
        f"Rules:\n- {rules_str}\n\n"
        "Input:\n{raw_log}\n\nJSON:"
    )


# Fields the prompt does NOT ask for — skip them in scoring
SKIP_FIELDS = {"EventId"}


def _norm_template(s: str) -> str:
    """Normalize a log event template: collapse <*> / * placeholders, strip whitespace."""
    s = re.sub(r"<\*+>|\*+", "<*>", s)  # any *-style placeholder → canonical <*>
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def fields_match(field: str, expected, actual, *, lenient: bool) -> bool:
    """Compare expected vs actual for a single field.

    Strict: full string equality after stripping.
    Lenient: per-field tolerance, with a generic fallback that handles any
    classification or extraction field (case-insensitive substring containment).
    """
    e = str(expected).strip()
    a = str(actual).strip()

    if e == a:
        return True
    if not lenient:
        return False

    # LogHub-specific tolerances
    if field == "Date":
        if a.lstrip("0") == e.lstrip("0"):
            return True
    if field == "EventTemplate":
        if _norm_template(a) == _norm_template(e):
            return True

    # Generic lenient fallback — works for ANY field (classification labels,
    # extraction substrings, log timestamps, etc.). Case-insensitive substring
    # containment in either direction.
    if a and e:
        a_lower = a.lower()
        e_lower = e.lower()
        if a_lower == e_lower:
            return True
        if a_lower in e_lower or e_lower in a_lower:
            return True

    return False


def evaluate_model(model, tokenizer, test_entries: list[dict], batch_size: int, parse_prompt: str) -> dict:
    """Score model on test set. Returns BOTH strict and lenient metrics."""
    n = len(test_entries)
    json_valid = 0
    per_entry: list[dict] = []

    # Per-field counters for strict + lenient
    field_total: dict[str, int] = {}
    strict_correct: dict[str, int] = {}
    lenient_correct: dict[str, int] = {}

    for start in range(0, n, batch_size):
        batch = test_entries[start:start + batch_size]
        prompts = [parse_prompt.format(raw_log=e["raw_log"]) for e in batch]
        responses = batch_ask(model, tokenizer, prompts, max_tokens=300)

        for entry, raw_response in zip(batch, responses):
            predicted = extract_json(raw_response)
            gt = entry["ground_truth"]

            if predicted is None:
                per_entry.append({
                    "raw_log": entry["raw_log"],
                    "ground_truth": gt,
                    "raw_response": raw_response[:400],
                    "predicted": None,
                })
                for field in gt:
                    if field in SKIP_FIELDS:
                        continue
                    field_total[field] = field_total.get(field, 0) + 1
                continue

            json_valid += 1
            for field, expected in gt.items():
                if field in SKIP_FIELDS:
                    continue
                field_total[field] = field_total.get(field, 0) + 1
                actual = predicted.get(field, "")
                if fields_match(field, expected, actual, lenient=False):
                    strict_correct[field] = strict_correct.get(field, 0) + 1
                if fields_match(field, expected, actual, lenient=True):
                    lenient_correct[field] = lenient_correct.get(field, 0) + 1
            per_entry.append({
                "raw_log": entry["raw_log"],
                "ground_truth": gt,
                "predicted": predicted,
            })

        print(f"  [{min(start + batch_size, n)}/{n}]", flush=True)

    def _fields_pct(correct: dict[str, int]) -> dict[str, float]:
        return {
            f: round(100 * correct.get(f, 0) / field_total[f], 1)
            for f in field_total
        }

    def _overall_pct(correct: dict[str, int]) -> float:
        return round(
            100 * sum(correct.values()) / max(sum(field_total.values()), 1),
            1,
        )

    return {
        "n": n,
        "json_valid_pct": round(100 * json_valid / n, 1),
        "strict": {
            "field_accuracy": _fields_pct(strict_correct),
            "overall_accuracy": _overall_pct(strict_correct),
        },
        "lenient": {
            "field_accuracy": _fields_pct(lenient_correct),
            "overall_accuracy": _overall_pct(lenient_correct),
        },
        "per_entry": per_entry,
    }


def print_summary(name: str, result: dict) -> None:
    print(f"\n--- {name} ---")
    print(f"  json valid: {result['json_valid_pct']}%")
    print(f"  STRICT  overall: {result['strict']['overall_accuracy']}%")
    print(f"  LENIENT overall: {result['lenient']['overall_accuracy']}%")
    print(f"  {'field':<18} {'strict':>8} {'lenient':>9}")
    for f in sorted(result["strict"]["field_accuracy"]):
        s = result["strict"]["field_accuracy"][f]
        l = result["lenient"]["field_accuracy"][f]
        print(f"    {f:<16} {s:>7}% {l:>8}%")


def print_comparison(results: dict[str, dict], mode: str = "lenient") -> None:
    """Print a side-by-side table for one mode (strict or lenient)."""
    fields = sorted({f for r in results.values() for f in r[mode]["field_accuracy"]})
    header = f"{'model':<20}" + "".join(f"{f[:12]:>14}" for f in fields) + f"{'overall':>14}{'json%':>10}"
    print("\n" + "=" * len(header))
    print(f"COMPARISON ({mode.upper()})")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        row = f"{name[:20]:<20}"
        for f in fields:
            row += f"{r[mode]['field_accuracy'].get(f, 0):>13.1f}%"
        row += f"{r[mode]['overall_accuracy']:>13.1f}%"
        row += f"{r['json_valid_pct']:>9.1f}%"
        print(row)
    print("=" * len(header))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-json", type=Path, required=True,
                    help="loghub_test.json from prep_loghub.py")
    ap.add_argument("--adapters", nargs="*", default=[], type=Path,
                    help="paths to LoRA adapter dirs (one per trained version)")
    ap.add_argument("--num", type=int, default=200,
                    help="how many test entries to score (random sample)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--output", type=Path, default=Path("./eval_loghub_results.json"))
    args = ap.parse_args()

    if "HF_HOME" not in os.environ:
        candidate = "/blue/dferris/y.sarwaiya/hf_cache"
        if os.path.isdir(os.path.dirname(candidate)):
            os.environ["HF_HOME"] = candidate

    # ── Load test set ────────────────────────────────────────────────
    test = json.loads(args.test_json.read_text())
    if args.num and args.num < len(test):
        random.Random(args.seed).shuffle(test)
        test = test[:args.num]
    print(f"📂 evaluating {len(test)} test entries from {args.test_json}", flush=True)

    # ── Derive the field set + label space from the test ground truth ──
    # Build the prompt around whichever fields the test set actually uses.
    # This lets the same script handle LogHub, CUAD, CRM, anything.
    # If a field has a small bounded label set (5-200 unique values, all
    # short strings), treat it as classification and include the labels
    # in the prompt as constraints. Without this, classification tasks
    # like CUAD score near-zero because the model has to guess CUAD's
    # exact 41-label vocabulary from nothing.
    field_counts: dict[str, int] = {}
    field_values: dict[str, set[str]] = {}
    for entry in test:
        for field, value in entry.get("ground_truth", {}).items():
            if field in SKIP_FIELDS:
                continue
            field_counts[field] = field_counts.get(field, 0) + 1
            if isinstance(value, str):
                field_values.setdefault(field, set()).add(value.strip())
    fields_to_extract = sorted(field_counts.keys())

    label_sets: dict[str, list[str]] = {}
    for field, values in field_values.items():
        n = len(values)
        max_len = max((len(v) for v in values), default=0)
        # Heuristic: if the field has a smallish bounded vocab of short values,
        # treat as classification. Avoid log-style fields where every value
        # is unique or long (timestamps, free text, etc.).
        if 2 <= n <= 200 and max_len <= 80:
            label_sets[field] = sorted(values)

    parse_prompt = build_parse_prompt(fields_to_extract, label_sets=label_sets)
    print(f"📋 fields to extract: {fields_to_extract}", flush=True)
    if label_sets:
        for f, labels in label_sets.items():
            print(f"📋 label set for '{f}': {len(labels)} options "
                  f"(showing first 5: {labels[:5]})", flush=True)

    # ── Tokenizer (shared across all evals) ──────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def load_fresh_base():
        """Load base from scratch. Used between adapter evals to avoid
        peft stacking — PeftModel.from_pretrained mutates the base, so we
        reload to guarantee clean weights for each adapter test."""
        print(f"\n🤖 loading base: {BASE_MODEL}", flush=True)
        m = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, device_map="auto",
        )
        m.eval()
        return m

    results: dict[str, dict] = {}

    # ── 1. Base model (no adapter) ────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVAL: base Llama 3.1 8B (untrained)")
    print("=" * 60)
    base = load_fresh_base()
    results["base"] = evaluate_model(base, tokenizer, test, args.batch_size, parse_prompt)
    print_summary("base", results["base"])
    del base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 2. Each adapter on a freshly-loaded base ─────────────────────
    for adapter_path in args.adapters:
        label = adapter_path.parent.name.replace("output_", "")
        short = label.split("_")[0] if "_" in label else label

        print("\n" + "=" * 60)
        print(f"EVAL: trained adapter {short}  ({adapter_path})")
        print("=" * 60)
        if not adapter_path.exists():
            print(f"  ⚠️  skipping — adapter not found: {adapter_path}")
            continue

        # Reload base fresh for each adapter — prevents peft stacking
        base = load_fresh_base()
        wrapped = PeftModel.from_pretrained(base, str(adapter_path))
        wrapped.eval()
        results[short] = evaluate_model(wrapped, tokenizer, test, args.batch_size, parse_prompt)
        print_summary(short, results[short])

        del wrapped, base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Comparison + save (both modes) ────────────────────────────────
    print_comparison(results, mode="strict")
    print_comparison(results, mode="lenient")

    summary = {
        name: {k: v for k, v in r.items() if k != "per_entry"}
        for name, r in results.items()
    }
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"\n📁 wrote {args.output}")

    detail_path = args.output.parent / (args.output.stem + "_detail.json")
    detail_path.write_text(json.dumps(results, indent=2))
    print(f"📁 wrote {detail_path}  (per-entry predictions for inspection)")


if __name__ == "__main__":
    main()
