"""CUAD → Aasaan-ready data.

Loads the CUAD legal contract dataset from HuggingFace and converts it into:

  1. cuad_train.csv      one column `clause_text`  → input to run.py
  2. cuad_test.json      list of {raw_log, ground_truth: {clause_type}}
                          (uses `raw_log` key to stay compatible with
                          evaluate_loghub.py — same eval script works)

CUAD is shaped as SQuAD-style QA. Each example has a `question` like
"Highlight the parts (if any) of this contract related to 'Anti-Assignment'..."
and `answers.text` containing the extracted clause text. We:
  - Keep only examples with non-empty answers (clauses that exist)
  - Extract the short label from the question (e.g. "Anti-Assignment")
  - Train/test split (different contracts in each split → no template leakage)

Usage:
    python prep_cuad.py --out-dir ./cuad_data
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path


def extract_label(question: str) -> str | None:
    """Pull the short clause-type label out of a CUAD question.

    CUAD questions look like:
      Highlight the parts (if any) of this contract related to "Anti-Assignment" that should be reviewed by a lawyer.

    The label is the quoted string. Returns None if no label can be extracted.
    """
    m = re.search(r'"([^"]+)"', question)
    if m:
        return m.group(1).strip()
    # Fallback: first 60 chars of the question
    return question[:60].strip() if question else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("./cuad_data"))
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CUAD from HuggingFace (theatticusproject/cuad-qa)...", flush=True)
    from datasets import load_dataset

    ds = load_dataset("theatticusproject/cuad-qa")
    train_split = ds["train"]
    print(f"  raw train examples: {len(train_split)}", flush=True)

    rows: list[dict[str, str]] = []
    skipped_no_answer = 0
    skipped_no_label = 0

    for ex in train_split:
        answers = ex.get("answers", {}).get("text", [])
        if not answers or not answers[0].strip():
            skipped_no_answer += 1
            continue

        label = extract_label(ex.get("question", ""))
        if not label:
            skipped_no_label += 1
            continue

        clause_text = answers[0].strip()
        if len(clause_text) < 30:
            continue

        rows.append({
            "clause_text": clause_text,
            "label": label,
            "contract": ex.get("title", ""),
        })

    print(
        f"  kept {len(rows)} clauses "
        f"(skipped {skipped_no_answer} with no answer, {skipped_no_label} with no label)",
        flush=True,
    )

    # Group by contract — split BY CONTRACT so test contracts are unseen during training
    by_contract: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_contract.setdefault(r["contract"], []).append(r)

    contracts = list(by_contract.keys())
    rng = random.Random(args.seed)
    rng.shuffle(contracts)

    n_test_contracts = max(1, len(contracts) // 5)  # ~20% of contracts → test
    test_contracts = set(contracts[:n_test_contracts])

    train_rows = [r for r in rows if r["contract"] not in test_contracts]
    test_rows = [r for r in rows if r["contract"] in test_contracts]

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    train_rows = train_rows[:args.n_train]
    test_rows = test_rows[:args.n_test]

    # Label distribution sanity
    train_labels = Counter(r["label"] for r in train_rows)
    test_labels = Counter(r["label"] for r in test_rows)
    print(f"\n  train labels: {len(train_labels)} unique, top 5: {train_labels.most_common(5)}")
    print(f"  test labels:  {len(test_labels)} unique, top 5: {test_labels.most_common(5)}")
    print(f"  test-only labels (not in train): "
          f"{[l for l in test_labels if l not in train_labels]}")

    # ── Write train CSV ───────────────────────────────────────────────
    train_csv = args.out_dir / "cuad_train.csv"
    with train_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clause_text", "label"])
        w.writeheader()
        for r in train_rows:
            w.writerow({"clause_text": r["clause_text"], "label": r["label"]})

    # ── Write test JSON ───────────────────────────────────────────────
    # Use `raw_log` key so evaluate_loghub.py can score it without modification.
    test_json = [
        {
            "raw_log": r["clause_text"],
            "system": "cuad",
            "ground_truth": {"clause_type": r["label"]},
        }
        for r in test_rows
    ]
    test_path = args.out_dir / "cuad_test.json"
    test_path.write_text(json.dumps(test_json, indent=2))

    print(f"\n📁 {train_csv} ({len(train_rows)} rows)")
    print(f"📁 {test_path} ({len(test_rows)} rows)")
    print(f"\nNext: sbatch run.sbatch {train_csv} clause_text 200 v3-refine")


if __name__ == "__main__":
    main()
