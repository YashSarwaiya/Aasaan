"""CUAD → Aasaan-ready data.

Downloads CUAD legal contract dataset from its official Zenodo release and
converts it into:

  1. cuad_train.csv      one column `clause_text`  → input to run.py
  2. cuad_test.json      list of {raw_log, ground_truth: {clause_type}}
                          (uses `raw_log` key to stay compatible with
                          evaluate_loghub.py — same eval script works)

CUAD ships in SQuAD JSON format. Each example has a question like
"Highlight the parts (if any) of this contract related to 'Anti-Assignment'..."
and `answers.text` containing the extracted clause text. We:
  - Keep only examples with non-empty answers (clauses that exist)
  - Extract the short label from the question (e.g. "Anti-Assignment")
  - Train/test split by contract → no contract leakage between splits

Usage:
    python prep_cuad.py --out-dir ./cuad_data

Why direct download vs HuggingFace `datasets`:
The HF `theatticusproject/cuad-qa` dataset uses a loader script which is
no longer supported in datasets >= 3.0. Direct Zenodo download is the
official source and avoids that dependency.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path


CUAD_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip"


def download_cuad(out_dir: Path) -> Path:
    """Download + extract CUAD_v1.json from Zenodo. Returns path to JSON.

    Caches the JSON locally — re-running won't re-download.
    """
    json_path = out_dir / "CUAD_v1.json"
    if json_path.exists():
        print(f"Using cached {json_path}", flush=True)
        return json_path

    print(f"Downloading CUAD zip from Zenodo (~9 MB compressed, ~50 MB extracted)...", flush=True)
    with urllib.request.urlopen(CUAD_URL) as resp:
        zip_bytes = resp.read()
    print(f"  downloaded {len(zip_bytes) / 1e6:.1f} MB", flush=True)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for name in z.namelist():
            if name.endswith("CUAD_v1.json"):
                print(f"  extracting {name}", flush=True)
                with z.open(name) as f:
                    json_path.write_bytes(f.read())
                break
        else:
            raise RuntimeError("CUAD_v1.json not found in the zip — Zenodo layout may have changed")

    return json_path


def extract_label(question: str) -> str | None:
    """Pull the short clause-type label out of a CUAD question.

    CUAD questions look like:
      Highlight the parts (if any) of this contract related to "Anti-Assignment" that should be reviewed by a lawyer.

    The label is the quoted string. Returns None if no label can be extracted.
    """
    m = re.search(r'"([^"]+)"', question)
    if m:
        return m.group(1).strip()
    return question[:60].strip() if question else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("./cuad_data"))
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    json_path = download_cuad(args.out_dir)
    print(f"Parsing {json_path}...", flush=True)
    data = json.loads(json_path.read_text())

    rows: list[dict[str, str]] = []
    skipped_no_answer = 0
    skipped_no_label = 0
    total_qas = 0

    for entry in data.get("data", []):
        title = entry.get("title", "")
        for para in entry.get("paragraphs", []):
            for qa in para.get("qas", []):
                total_qas += 1
                answers = qa.get("answers", [])
                if not answers:
                    skipped_no_answer += 1
                    continue

                clause_text = (answers[0].get("text") or "").strip()
                if len(clause_text) < 30:
                    skipped_no_answer += 1
                    continue

                label = extract_label(qa.get("question", ""))
                if not label:
                    skipped_no_label += 1
                    continue

                rows.append({
                    "clause_text": clause_text,
                    "label": label,
                    "contract": title,
                })

    print(
        f"  total QAs: {total_qas} | kept {len(rows)} clauses "
        f"(skipped {skipped_no_answer} empty/short, {skipped_no_label} unlabeled)",
        flush=True,
    )
    if not rows:
        raise SystemExit("no usable clauses extracted — check CUAD JSON structure")

    # Split BY CONTRACT so test contracts are unseen during training
    by_contract: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_contract.setdefault(r["contract"], []).append(r)

    contracts = list(by_contract.keys())
    rng = random.Random(args.seed)
    rng.shuffle(contracts)

    n_test_contracts = max(1, len(contracts) // 5)
    test_contracts = set(contracts[:n_test_contracts])
    print(f"  contracts: {len(contracts)} total, {n_test_contracts} held out for test", flush=True)

    train_rows = [r for r in rows if r["contract"] not in test_contracts]
    test_rows = [r for r in rows if r["contract"] in test_contracts]

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    train_rows = train_rows[:args.n_train]
    test_rows = test_rows[:args.n_test]

    train_labels = Counter(r["label"] for r in train_rows)
    test_labels = Counter(r["label"] for r in test_rows)
    print(f"\n  train labels: {len(train_labels)} unique, top 5: {train_labels.most_common(5)}")
    print(f"  test labels:  {len(test_labels)} unique, top 5: {test_labels.most_common(5)}")
    test_only = [l for l in test_labels if l not in train_labels]
    if test_only:
        print(f"  test-only labels (not in train): {test_only}")

    train_csv = args.out_dir / "cuad_train.csv"
    with train_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clause_text", "label"])
        w.writeheader()
        for r in train_rows:
            w.writerow({"clause_text": r["clause_text"], "label": r["label"]})

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
