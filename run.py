"""
aasaan — standalone HiPerGator CLI.

Takes a messy file in (CSV, PDF, TXT), produces:
  - domain.txt              detected domain ("medical clinical notes")
  - schema.json             auto-generated 12-field schema
  - questions.json          10 generated questions
  - structured.json         extracted structured forms
  - training_data_v2.json   Q&A pairs (real LLM answers)
  - training_data_clean.json filtered training data (drops "Not specified" junk)
  - adapter/                final LoRA adapter (with --train)
  - checkpoints/            intermediate checkpoints (every 50 steps)

Composes the pure-Python modules in pipeline/. Same modules run on Modal in
the productized version, so behavior is identical.

Usage (typical):
    python run.py --input mtsamples.csv --num 250 --output ./out --train

Iterate-on-schema workflow:
    python run.py --input data.csv --output ./out --skip-train       # data prep
    # inspect ./out/structured.json — does it look right?
    python run.py --output ./out --train-only                        # just train

SLURM:
    sbatch run.sbatch mtsamples.csv transcription 250
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path

from pipeline import extract, llm, qa, schema, train


def _eprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ── Input loaders ────────────────────────────────────────────────────────


def load_documents(input_path: Path, column: str | None, num: int) -> list[str]:
    """Read input file → list of documents.

    CSV:  reads `column` (default 'transcription'), drops empty rows, samples N.
    PDF:  one doc per page.
    TXT:  splits on blank lines.
    """
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(input_path)
        col = column or _pick_text_column(df)
        df = df.dropna(subset=[col])
        df = df.sample(n=min(num, len(df)), random_state=42).reset_index(drop=True)
        docs = df[col].tolist()
        _eprint(f"📂 Loaded {len(docs)} CSV rows from column '{col}'")
        return docs

    if suffix == ".pdf":
        from pypdf import PdfReader

        with open(input_path, "rb") as f:
            reader = PdfReader(io.BytesIO(f.read()))
            docs = [p.extract_text() or "" for p in reader.pages]
        docs = [d.strip() for d in docs if len(d.strip()) > 50][:num]
        _eprint(f"📂 Loaded {len(docs)} PDF pages")
        return docs

    # .txt or anything else → one doc per blank-line block
    text = input_path.read_text(encoding="utf-8", errors="ignore")
    docs = [b.strip() for b in text.split("\n\n") if len(b.strip()) > 50][:num]
    _eprint(f"📂 Loaded {len(docs)} text blocks")
    return docs


def _pick_text_column(df) -> str:
    """If --column wasn't passed, pick the longest-string column."""
    preferred = ("transcription", "text", "content", "body", "note", "description")
    cols = list(df.columns)
    for p in preferred:
        if p in cols:
            return p
    # Fallback: column with longest average string length
    return max(cols, key=lambda c: sum(len(str(v)) for v in df[c].head(50)))


# ── Phase runners ────────────────────────────────────────────────────────


def run_data_prep(
    input_path: Path,
    output_dir: Path,
    column: str | None,
    num: int,
    extract_batch_size: int,
    qa_batch_size: int,
) -> dict:
    """Phases 1-8: load → domain → schema → questions → extract → Q&A → filter.

    Saves all artifacts under output_dir. Returns metadata about the run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    docs = load_documents(input_path, column, num)
    if not docs:
        raise SystemExit("no documents loaded — check --input and --column")

    _eprint("\n🤖 Loading Qwen 2.5 7B Instruct...")
    model, tokenizer = llm.load_model("Qwen/Qwen2.5-7B-Instruct")
    _eprint("✅ Model loaded\n")

    # ── 1. Detect domain ─────────────────────────────────────────────
    _eprint("=" * 60)
    _eprint("STEP 1: 🔍 detecting domain")
    _eprint("=" * 60)
    domain = schema.detect_domain(model, tokenizer, docs)
    _eprint(f"✅ {domain}\n")
    (output_dir / "domain.txt").write_text(domain)

    # ── 2. Build schema ──────────────────────────────────────────────
    _eprint("=" * 60)
    _eprint("STEP 2: 📋 building schema")
    _eprint("=" * 60)
    sch = schema.build_schema(model, tokenizer, domain, docs)
    _eprint(f"✅ {len(sch)} fields\n{json.dumps(sch, indent=2)[:600]}\n")
    with open(output_dir / "schema.json", "w") as f:
        json.dump({"domain": domain, "schema": sch}, f, indent=2)

    # ── 3. Generate questions ────────────────────────────────────────
    _eprint("=" * 60)
    _eprint("STEP 3: ❓ generating questions")
    _eprint("=" * 60)
    questions = schema.generate_questions(model, tokenizer, domain, sch)
    _eprint(f"✅ {len(questions)} questions")
    for q in questions[:5]:
        _eprint(f"  - {q}")
    _eprint("")
    with open(output_dir / "questions.json", "w") as f:
        json.dump(questions, f, indent=2)

    # ── 4. Extract structured ────────────────────────────────────────
    _eprint("=" * 60)
    _eprint(f"STEP 4: 🔄 extracting structured ({len(docs)} docs)")
    _eprint("=" * 60)

    def on_extract(done: int, total: int):
        if done % (extract_batch_size * 4) == 0:
            _eprint(f"  ...{done}/{total}")

    structured, failed = extract.batch_extract_structured(
        model,
        tokenizer,
        docs,
        domain,
        sch,
        batch_size=extract_batch_size,
        on_progress=on_extract,
    )
    _eprint(f"✅ {len(structured)} extracted, {failed} skipped\n")
    with open(output_dir / "structured.json", "w") as f:
        json.dump(structured, f, indent=2)

    # ── 5. Generate Q&A ──────────────────────────────────────────────
    _eprint("=" * 60)
    _eprint("STEP 5: 🧠 generating training Q&A")
    _eprint("=" * 60)

    def on_qa(done: int, total: int):
        if done % (qa_batch_size * 4) == 0:
            _eprint(f"  ...{done}/{total}")

    training_data = qa.generate_qa(
        model,
        tokenizer,
        structured,
        questions,
        domain,
        batch_size=qa_batch_size,
        on_progress=on_qa,
    )
    _eprint(f"✅ {len(training_data)} Q&A pairs\n")
    with open(output_dir / "training_data_v2.json", "w") as f:
        json.dump(training_data, f, indent=2)

    # ── 6. Filter clean ──────────────────────────────────────────────
    _eprint("=" * 60)
    _eprint("STEP 6: 🧹 filtering noisy rows")
    _eprint("=" * 60)
    clean = qa.filter_clean(training_data)
    _eprint(f"✅ {len(clean)} clean (dropped {len(training_data) - len(clean)})\n")
    with open(output_dir / "training_data_clean.json", "w") as f:
        json.dump(clean, f, indent=2)

    elapsed = time.time() - started
    metadata = {
        "domain": domain,
        "num_input_docs": len(docs),
        "schema_fields": len(sch),
        "structured_count": len(structured),
        "extraction_failed": failed,
        "training_pairs": len(training_data),
        "clean_pairs": len(clean),
        "data_prep_seconds": round(elapsed, 1),
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    _eprint(f"⏱️  data prep took {elapsed:.0f}s")
    return metadata


def run_training(output_dir: Path, epochs: int) -> dict:
    """Phase 9: load training_data_clean.json from output_dir and run LoRA.

    Final adapter goes to <output_dir>/adapter/.
    Checkpoints to <output_dir>/checkpoints/checkpoint-N/ (saved every 50 steps).
    """
    clean_path = output_dir / "training_data_clean.json"
    if not clean_path.exists():
        raise SystemExit(
            f"no training data at {clean_path} — run data prep first "
            f"(re-run without --train-only)"
        )
    with open(clean_path) as f:
        clean = json.load(f)

    if len(clean) < 20:
        raise SystemExit(f"only {len(clean)} clean pairs — need at least 20")

    _eprint("\n🤖 Loading Qwen 2.5 7B Instruct for fine-tuning...")
    model, tokenizer = llm.load_model("Qwen/Qwen2.5-7B-Instruct")
    _eprint("✅ Model loaded\n")

    _eprint("=" * 60)
    _eprint(f"STEP 7: 🚀 fine-tuning LoRA on {len(clean)} clean pairs")
    _eprint("=" * 60)

    started = time.time()
    adapter_dir = output_dir / "adapter"

    def on_progress(pct: float, msg: str):
        _eprint(f"  [{pct:.0f}%] {msg}")

    train.train_lora(
        model,
        tokenizer,
        clean,
        str(adapter_dir),
        on_progress=on_progress,
        commit_volume=None,        # no Modal volume on HiPerGator — local disk only
        on_checkpoint=lambda step, path: _eprint(f"  💾 checkpoint @ step {step}"),
        epochs=epochs,
    )
    elapsed = time.time() - started
    _eprint(f"\n✅ Adapter saved to {adapter_dir}")
    _eprint(f"⏱️  training took {elapsed/60:.1f} min")

    # Update metadata if it exists
    meta_path = output_dir / "run_metadata.json"
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    metadata["training_seconds"] = round(elapsed, 1)
    metadata["epochs"] = epochs
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="aasaan standalone pipeline (HiPerGator-friendly)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, help="path to CSV/PDF/TXT")
    parser.add_argument(
        "--column",
        default=None,
        help="text column for CSVs (auto-detected if omitted)",
    )
    parser.add_argument("--num", type=int, default=250, help="docs to sample")
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--qa-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--skip-train",
        action="store_true",
        help="run data prep only, skip LoRA training",
    )
    mode.add_argument(
        "--train-only",
        action="store_true",
        help="skip data prep, just train on existing training_data_clean.json",
    )
    mode.add_argument(
        "--train",
        action="store_true",
        help="run data prep then train (the typical full pipeline)",
    )

    args = parser.parse_args()

    if args.train_only:
        run_training(args.output, args.epochs)
        return

    if not args.input:
        raise SystemExit("--input is required unless --train-only")

    run_data_prep(
        input_path=args.input,
        output_dir=args.output,
        column=args.column,
        num=args.num,
        extract_batch_size=args.extract_batch_size,
        qa_batch_size=args.qa_batch_size,
    )

    if args.train:
        run_training(args.output, args.epochs)

    _eprint("\n🎉 Done.")
    _eprint(f"📁 {args.output}")


if __name__ == "__main__":
    # Default HF cache to scratch on HiPerGator. Caller can override via env.
    if "HF_HOME" not in os.environ:
        candidate = "/blue/dferris/y.sarwaiya/hf_cache"
        if os.path.isdir(os.path.dirname(candidate)):
            os.environ["HF_HOME"] = candidate
    main()
