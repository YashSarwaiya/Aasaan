"""LogHub → Aasaan-ready data.

Produces TWO artifacts (both essential):

  1. loghub_train.csv      one column `raw_log`  → input to run.py
  2. loghub_test.json      list of {raw_log, system, ground_truth: {...}}
                           → eval against LogHub's structured labels

Why two? `raw_log` alone is enough to GENERATE training data (run.py auto-builds
a schema and Q&A from messy text). But measuring "did fine-tuning improve
accuracy" needs labels. LogHub already ships parsed labels in
`<sys>_2k.log_structured.csv` — we just zip them with the raw lines.

Splits by EventTemplate (NOT by line). Held-out templates ensure the test set
measures generalization, not memorization. Train/test counts are approximate.

Usage:
    git clone https://github.com/logpai/loghub.git /blue/dferris/y.sarwaiya/loghub
    python prep_loghub.py \
        --loghub-root /blue/dferris/y.sarwaiya/loghub \
        --systems HDFS Apache Linux \
        --out-dir ./loghub_data
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


DEFAULT_SYSTEMS = ["HDFS", "Apache", "Linux"]
TRAIN_CAP = 1700      # per system
TEST_TARGET = 300     # per system (approximate — depends on template sizes)
GROUND_TRUTH_FIELDS = ["Date", "Time", "Level", "Component", "Content", "EventTemplate", "EventId"]


def load_system(loghub_root: Path, sys_name: str) -> pd.DataFrame:
    """Read one system's raw log + structured labels, joined by LineId."""
    log_path = loghub_root / sys_name / f"{sys_name}_2k.log"
    csv_path = loghub_root / sys_name / f"{sys_name}_2k.log_structured.csv"

    if not log_path.exists():
        raise SystemExit(f"missing raw log: {log_path}")
    if not csv_path.exists():
        raise SystemExit(f"missing structured csv: {csv_path}")

    raw_lines = log_path.read_text(errors="replace").splitlines()
    df = pd.read_csv(csv_path)

    if "LineId" not in df.columns:
        raise SystemExit(f"{csv_path} missing LineId column")

    df["raw_log"] = df["LineId"].apply(lambda lid: raw_lines[int(lid) - 1] if 0 < int(lid) <= len(raw_lines) else "")
    df["system"] = sys_name
    return df


def split_by_template(df: pd.DataFrame, rng: random.Random, test_target: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole EventIds (templates) for the test set, total ~test_target lines."""
    if "EventId" not in df.columns:
        # Some systems may not have EventId — fall back to random line split
        shuffled = df.sample(frac=1.0, random_state=rng.randint(0, 1_000_000)).reset_index(drop=True)
        return shuffled.iloc[test_target:], shuffled.iloc[:test_target]

    eids = list(df["EventId"].unique())
    rng.shuffle(eids)

    held_out: list[str] = []
    accumulated = 0
    for eid in eids:
        n = int((df["EventId"] == eid).sum())
        if accumulated >= test_target:
            break
        held_out.append(eid)
        accumulated += n

    test_df = df[df["EventId"].isin(held_out)].copy()
    train_df = df[~df["EventId"].isin(held_out)].copy()
    return train_df, test_df


def to_ground_truth(row: pd.Series) -> dict:
    """Pull the parseable fields LogHub provides as ground truth."""
    gt = {}
    for field in GROUND_TRUTH_FIELDS:
        if field in row.index and pd.notna(row[field]):
            gt[field] = str(row[field])
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loghub-root", type=Path, required=True,
                    help="path to a cloned logpai/loghub repo")
    ap.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS,
                    help="systems to include (subdir names under loghub-root)")
    ap.add_argument("--out-dir", type=Path, default=Path("./loghub_data"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-cap", type=int, default=TRAIN_CAP,
                    help="max train lines per system (after template split)")
    ap.add_argument("--test-target", type=int, default=TEST_TARGET,
                    help="approximate test lines per system (whole templates held out)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    train_rows: list[dict] = []
    test_rows: list[dict] = []
    summary: list[str] = []

    for sys_name in args.systems:
        df = load_system(args.loghub_root, sys_name)
        train_df, test_df = split_by_template(df, rng, args.test_target)

        # Cap training at the per-system limit (random within remaining templates)
        if len(train_df) > args.train_cap:
            train_df = train_df.sample(args.train_cap, random_state=args.seed).reset_index(drop=True)

        for _, row in train_df.iterrows():
            train_rows.append({"raw_log": row["raw_log"], "system": sys_name})

        for _, row in test_df.iterrows():
            test_rows.append({
                "raw_log": row["raw_log"],
                "system": sys_name,
                "ground_truth": to_ground_truth(row),
            })

        summary.append(
            f"  {sys_name}: {len(train_df)} train / {len(test_df)} test "
            f"(test holds out {df['EventId'].nunique() - train_df['EventId'].nunique() if 'EventId' in df.columns else '?'} templates)"
        )

    # Shuffle so systems are interleaved (avoids batch-effect bias during training)
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    # ── Write training CSV (input to run.py) ──────────────────────────
    train_csv = args.out_dir / "loghub_train.csv"
    pd.DataFrame(train_rows).to_csv(train_csv, index=False)

    # ── Write test JSON (with ground truth, for evaluation) ───────────
    test_json = args.out_dir / "loghub_test.json"
    test_json.write_text(json.dumps(test_rows, indent=2))

    print("\n=== LogHub data prep complete ===")
    for line in summary:
        print(line)
    print(f"\n📁 {train_csv} ({len(train_rows)} rows)")
    print(f"📁 {test_json} ({len(test_rows)} rows)")
    print(f"\nNext: python run.py --input {train_csv} --column raw_log --num 500")


if __name__ == "__main__":
    main()
