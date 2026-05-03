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

PARSE_PROMPT = """You are a log parser. Parse this log line into JSON.

Output ONLY valid JSON with these fields if present (omit any not in the line):
- Date, Time, Level, Component, Content, EventTemplate

Rules:
- Use the EXACT substring from the log. Do NOT reformat dates or paraphrase content.
- For EventTemplate, replace variable parts (numbers, IDs, paths, blocks) with <*>.
- No explanations, no markdown, no preamble. JSON only.

Log line:
{raw_log}

JSON:"""


# Fields the prompt does NOT ask for — skip them in scoring (they'd be 0% by construction)
SKIP_FIELDS = {"EventId"}


def _norm_template(s: str) -> str:
    """Normalize a log event template: collapse <*> / * placeholders, strip whitespace."""
    s = re.sub(r"<\*+>|\*+", "<*>", s)  # any *-style placeholder → canonical <*>
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def fields_match(field: str, expected, actual, *, lenient: bool) -> bool:
    """Compare expected vs actual for a single field.

    Strict: full string equality after stripping.
    Lenient: per-field tolerance for legitimate format variants
      - Date: ignore leading zeros (CSV import stripped them from GT)
      - Date / Time: substring containment in either direction
      - Component / Content: substring containment in either direction
      - EventTemplate: normalize placeholders + whitespace, case-insensitive
      - Level: case-insensitive
    """
    e = str(expected).strip()
    a = str(actual).strip()

    if e == a:
        return True
    if not lenient:
        return False

    if field == "Date":
        if a.lstrip("0") == e.lstrip("0"):
            return True
        if a in e or e in a:
            return True
    if field == "Time":
        if a in e or e in a:
            return True
    if field == "Component" or field == "Content":
        if a and e and (a in e or e in a):
            return True
    if field == "EventTemplate":
        if _norm_template(a) == _norm_template(e):
            return True
    if field == "Level":
        if a.lower() == e.lower():
            return True
    return False


def evaluate_model(model, tokenizer, test_entries: list[dict], batch_size: int) -> dict:
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
        prompts = [PARSE_PROMPT.format(raw_log=e["raw_log"]) for e in batch]
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
    print(f"📂 evaluating {len(test)} test logs from {args.test_json}", flush=True)

    # ── Load base + tokenizer (shared across all evals) ──────────────
    print(f"\n🤖 loading base: {BASE_MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="auto",
    )
    base.eval()
    print("✅ base loaded", flush=True)

    results: dict[str, dict] = {}

    # ── 1. Base model (no adapter) ────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVAL: base Llama 3.1 8B (untrained)")
    print("=" * 60)
    results["base"] = evaluate_model(base, tokenizer, test, args.batch_size)
    print_summary("base", results["base"])

    # ── 2. Each adapter, hot-swapped onto the same base ──────────────
    for adapter_path in args.adapters:
        # adapter_path is like "output_v3_TIMESTAMP/adapter" — use parent name as label
        label = adapter_path.parent.name.replace("output_", "")  # e.g. "v3_20260503..."
        # Trim trailing timestamp for cleaner table
        short = label.split("_")[0] if "_" in label else label  # "v3"

        print("\n" + "=" * 60)
        print(f"EVAL: trained adapter {short}  ({adapter_path})")
        print("=" * 60)
        if not adapter_path.exists():
            print(f"  ⚠️  skipping — adapter not found: {adapter_path}")
            continue

        wrapped = PeftModel.from_pretrained(base, str(adapter_path))
        wrapped.eval()
        results[short] = evaluate_model(wrapped, tokenizer, test, args.batch_size)
        print_summary(short, results[short])

        # Unload adapter so the next iteration sees the bare base again
        del wrapped
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
