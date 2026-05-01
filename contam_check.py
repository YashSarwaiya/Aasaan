"""Eval contamination probe.

Tests whether Qwen 2.5 7B has memorized MTSamples (the public clinical-notes
dataset our eval uses). If it has, our baseline numbers are inflated and the
"trained vs baseline" gap is misleading.

Three tests:
  1. Recognition — does Qwen know what MTSamples is?
  2. Completion — give it a real MTSamples opening, see if it auto-completes
     text that matches the actual note (vs hallucinating).
  3. Structural recall — ask for typical chief complaints in MTSamples;
     compare against the actual distribution.

Run as:
    sbatch --time=00:30:00 --partition=hpg-b200 --gres=gpu:1 \
        --account=dferris --qos=dferris --mem=64gb \
        --output=logs/contam-%j.out \
        --wrap="module load conda && conda activate doctorai && \
               cd /blue/dferris/y.sarwaiya/aasaan && \
               python contam_check.py --csv mtsample/mtsamples.csv"

Outputs:
    contam_results.json  — full text of each probe response
    contam_summary.txt   — human-readable verdict per test
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from pipeline.llm import ask, load_model


def test_recognition(model, tok) -> dict:
    """Test 1: ask Qwen if it knows MTSamples by name."""
    prompt = (
        "Have you seen the MTSamples medical transcription dataset before? "
        "If yes, briefly describe what it contains and where it's hosted. "
        "If you haven't seen it, say so clearly. Two sentences max."
    )
    response = ask(model, tok, prompt, max_tokens=200)
    contaminated = any(
        kw in response.lower()
        for kw in ("mtsamples", "transcription", "mtsample", "yes")
    )
    return {
        "test": "recognition",
        "prompt": prompt,
        "response": response,
        "contaminated_signal": contaminated,
    }


def test_completion(model, tok, real_note: str) -> dict:
    """Test 2: give Qwen the first 200 chars of a real MTSamples note,
    ask it to complete. Compare against the actual continuation."""
    prefix = real_note[:200]
    actual_continuation = real_note[200:600]
    prompt = (
        "Complete this clinical note exactly as it would appear in a "
        "medical transcription database. Continue from where it stops:\n\n"
        f"{prefix}"
    )
    response = ask(model, tok, prompt, max_tokens=200)

    # Token-overlap heuristic: high overlap = likely memorized
    actual_words = set(re.findall(r"\b\w{4,}\b", actual_continuation.lower()))
    response_words = set(re.findall(r"\b\w{4,}\b", response.lower()))
    overlap = len(actual_words & response_words)
    overlap_pct = overlap / max(len(actual_words), 1) * 100

    return {
        "test": "completion",
        "prefix": prefix,
        "actual_continuation": actual_continuation,
        "model_completion": response,
        "word_overlap_pct": round(overlap_pct, 1),
        "contaminated_signal": overlap_pct > 35,  # >35% rare-word overlap is suspicious
    }


def test_structural_recall(model, tok, real_complaints: list[str]) -> dict:
    """Test 3: ask Qwen for typical MTSamples chief complaints. Compare
    distribution to actual."""
    prompt = (
        "List 8 chief complaints that commonly appear in the MTSamples "
        "medical transcription dataset. Output as a simple list, one per line. "
        "Be specific and concise."
    )
    response = ask(model, tok, prompt, max_tokens=300)

    listed = [
        line.strip().lstrip("-•0123456789. )").strip().lower()
        for line in response.split("\n")
        if line.strip()
    ]
    listed = [l for l in listed if 3 < len(l) < 80][:8]

    real_lower = [c.lower() for c in real_complaints]
    matches = 0
    for item in listed:
        # Loose match: any 3+ char word from `item` appears in any real complaint
        words = re.findall(r"\b\w{4,}\b", item)
        for w in words:
            if any(w in rc for rc in real_lower):
                matches += 1
                break

    return {
        "test": "structural_recall",
        "prompt": prompt,
        "response": response,
        "model_listed": listed,
        "matches_real": matches,
        "contaminated_signal": matches >= 5,  # 5+/8 matches = likely seen the data
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--column", default="transcription")
    ap.add_argument(
        "--complaint-col",
        default=None,
        help="Column with chief complaints (auto-detected if omitted)",
    )
    ap.add_argument("--output", type=Path, default=Path("contam_results.json"))
    ap.add_argument(
        "--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model under test"
    )
    args = ap.parse_args()

    print(f"📂 Loading {args.csv}", flush=True)
    df = pd.read_csv(args.csv).dropna(subset=[args.column])

    # Pick a random real note for completion test
    real_note = df.sample(1, random_state=12345).iloc[0][args.column]

    # Pull complaints if available
    complaint_col = args.complaint_col
    if complaint_col is None:
        for cand in ("medical_specialty", "description", "chief_complaint"):
            if cand in df.columns:
                complaint_col = cand
                break
    if complaint_col and complaint_col in df.columns:
        real_complaints = df[complaint_col].dropna().head(50).tolist()
    else:
        real_complaints = []

    print(f"🤖 Loading {args.model}", flush=True)
    model, tok = load_model(args.model)

    results = {"model": args.model, "tests": []}

    print("\n=== TEST 1: recognition ===", flush=True)
    r1 = test_recognition(model, tok)
    print(f"Response: {r1['response'][:200]}")
    print(f"Contaminated signal: {r1['contaminated_signal']}")
    results["tests"].append(r1)

    print("\n=== TEST 2: completion ===", flush=True)
    r2 = test_completion(model, tok, real_note)
    print(f"Word overlap with actual continuation: {r2['word_overlap_pct']}%")
    print(f"Contaminated signal: {r2['contaminated_signal']}")
    results["tests"].append(r2)

    if real_complaints:
        print("\n=== TEST 3: structural recall ===", flush=True)
        r3 = test_structural_recall(model, tok, real_complaints)
        print(f"Model listed {len(r3['model_listed'])} complaints, "
              f"{r3['matches_real']} matched real distribution")
        print(f"Contaminated signal: {r3['contaminated_signal']}")
        results["tests"].append(r3)

    # Verdict
    contaminated_count = sum(1 for t in results["tests"] if t.get("contaminated_signal"))
    if contaminated_count >= 2:
        verdict = "LIKELY CONTAMINATED — eval numbers are biased; need uncontaminated test set"
    elif contaminated_count == 1:
        verdict = "AMBIGUOUS — one signal positive; investigate manually"
    else:
        verdict = "CLEAN — no strong contamination signal; eval numbers are likely valid"
    results["verdict"] = verdict

    args.output.write_text(json.dumps(results, indent=2))
    print(f"\n📁 wrote {args.output}")
    print(f"\n=== VERDICT ===")
    print(verdict)


if __name__ == "__main__":
    import os
    if "HF_HOME" not in os.environ:
        candidate = "/blue/dferris/y.sarwaiya/hf_cache"
        if os.path.isdir(os.path.dirname(candidate)):
            os.environ["HF_HOME"] = candidate
    main()
