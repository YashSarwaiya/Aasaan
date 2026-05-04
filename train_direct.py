"""Direct supervised fine-tuning baseline.

Strips the pipeline. Reads a CSV with two columns (input + output), converts
to Alpaca JSON, and trains LoRA — no auto-schema, no curriculum, no refine,
no DPO. The "moat question" baseline.

Used to answer: does our pipeline actually beat trivial supervised fine-tuning?
If direct SFT matches our pipeline's accuracy, the curriculum's complexity
isn't earning its keep.

Usage:
    python train_direct.py \
        --input ./cuad_data/cuad_train.csv \
        --input-col clause_text \
        --output-col label \
        --instruction "Classify this contract clause." \
        --num 200 \
        --output ./output_direct_cuad_200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from pipeline import llm, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="CSV with input + output columns")
    ap.add_argument("--input-col", required=True, help="column name for the input text")
    ap.add_argument("--output-col", required=True, help="column name for the target label/output")
    ap.add_argument("--instruction", required=True,
                    help="natural-language instruction (e.g. 'Classify this contract clause.')")
    ap.add_argument("--num", type=int, default=200, help="rows to sample (random_state=42)")
    ap.add_argument("--output", type=Path, required=True, help="output directory")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # ── Read CSV, sample, convert to Alpaca format ────────────────────
    print(f"📂 reading {args.input}", flush=True)
    df = pd.read_csv(args.input)
    if args.input_col not in df.columns or args.output_col not in df.columns:
        raise SystemExit(
            f"missing columns. got {list(df.columns)}, "
            f"need {args.input_col!r} and {args.output_col!r}"
        )
    df = df.dropna(subset=[args.input_col, args.output_col])
    if args.num and args.num < len(df):
        df = df.sample(n=args.num, random_state=42).reset_index(drop=True)
    print(f"  using {len(df)} rows", flush=True)

    pairs = [
        {
            "instruction": args.instruction,
            "input": str(row[args.input_col]),
            "output": str(row[args.output_col]),
        }
        for _, row in df.iterrows()
    ]
    clean_path = args.output / "training_data_clean.json"
    clean_path.write_text(json.dumps(pairs, indent=2))
    print(f"📁 wrote {clean_path} ({len(pairs)} pairs)", flush=True)

    # ── Train LoRA — same trainer the pipeline uses ──────────────────
    print(f"\n🤖 loading Llama 3.1 8B Instruct...", flush=True)
    model, tokenizer = llm.load_model("meta-llama/Llama-3.1-8B-Instruct")
    print("✅ model loaded\n", flush=True)

    adapter_dir = args.output / "adapter"
    print("=" * 60)
    print(f"🚀 direct SFT on {len(pairs)} pairs (NO pipeline)")
    print("=" * 60)
    started = time.time()

    def on_progress(pct: float, msg: str):
        print(f"  [{pct:.0f}%] {msg}", flush=True)

    train.train_lora(
        model,
        tokenizer,
        pairs,
        str(adapter_dir),
        on_progress=on_progress,
        commit_volume=None,
        on_checkpoint=lambda step, path: print(f"  💾 checkpoint @ step {step}", flush=True),
        epochs=args.epochs,
    )
    elapsed = time.time() - started
    print(f"\n✅ Adapter saved to {adapter_dir}")
    print(f"⏱️  training took {elapsed/60:.1f} min")

    # ── Save metadata so eval scripts can find it ────────────────────
    meta = {
        "method": "direct_sft",
        "input_csv": str(args.input),
        "input_col": args.input_col,
        "output_col": args.output_col,
        "instruction": args.instruction,
        "num_pairs": len(pairs),
        "epochs": args.epochs,
        "training_seconds": round(elapsed, 1),
    }
    (args.output / "run_metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    import os
    if "HF_HOME" not in os.environ:
        candidate = "/blue/dferris/y.sarwaiya/hf_cache"
        if os.path.isdir(os.path.dirname(candidate)):
            os.environ["HF_HOME"] = candidate
    main()
