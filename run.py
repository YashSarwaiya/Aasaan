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

from pipeline import (
    curriculum, dedup, dpo, extract, llm, pii, qa,
    quality, refine, schema, train,
)


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
    teacher_model_name: str = "Qwen/Qwen2.5-32B-Instruct",
) -> dict:
    """Multi-phase pipeline: load → domain → schema (with field split) → extract
    → multi-task curriculum (Qwen 32B teacher) → grounded validation (same 32B)
    → final clean training set.

    Saves all artifacts under output_dir. Returns metadata about the run.

    Two model loads:
      1. Qwen 7B for domain/schema/extraction (steps 1-4)
      2. Qwen 32B for curriculum + validation (steps 5-6)
      Then 7B reloaded for training in run_training().
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    docs = load_documents(input_path, column, num)
    if not docs:
        raise SystemExit("no documents loaded — check --input and --column")

    # ── Pre-pipeline data quality steps ──────────────────────────────
    # Both are no-ops on missing optional deps (datasketch / presidio).
    _eprint("\n=" * 60)
    _eprint("STEP 0: 🧹 dedup + PII stripping")
    _eprint("=" * 60)
    docs, n_dropped_dups = dedup.deduplicate(docs, threshold=0.85)
    docs, n_pii_redacted = pii.anonymize_documents(docs)

    _eprint(f"\n🤖 Loading Qwen 2.5 7B Instruct (extractor)...")
    model, tokenizer = llm.load_model("Qwen/Qwen2.5-7B-Instruct")
    _eprint("✅ Extractor loaded\n")

    # ── 1. Detect domain ─────────────────────────────────────────────
    _eprint("=" * 60)
    _eprint("STEP 1: 🔍 detecting domain")
    _eprint("=" * 60)
    domain = schema.detect_domain(model, tokenizer, docs)
    _eprint(f"✅ {domain}\n")
    (output_dir / "domain.txt").write_text(domain)

    # ── 2. Build schema + classify lookup vs reasoning ───────────────
    _eprint("=" * 60)
    _eprint("STEP 2: 📋 building schema + classifying fields")
    _eprint("=" * 60)
    sch = schema.build_schema(model, tokenizer, domain, docs)
    lookup_fields, reasoning_fields = schema.classify_fields(sch)
    _eprint(f"✅ {len(sch)} fields total")
    _eprint(f"   lookup ({len(lookup_fields)}, RAG-served): {lookup_fields}")
    _eprint(f"   reasoning ({len(reasoning_fields)}, train target): {reasoning_fields}\n")
    with open(output_dir / "schema.json", "w") as f:
        json.dump(
            {
                "domain": domain,
                "schema": sch,
                "lookup_fields": lookup_fields,
                "reasoning_fields": reasoning_fields,
            },
            f,
            indent=2,
        )

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

    # ── Free Qwen 7B before loading the 32B teacher ──────────────────
    _eprint("🧹 Freeing 7B from GPU before loading 32B teacher...")
    llm.unload_model(model)
    del tokenizer
    _eprint("✅ Memory cleared\n")

    # ── 5. Multi-task curriculum (Qwen 32B teacher) ──────────────────
    _eprint("=" * 60)
    _eprint(f"STEP 5: 🎓 multi-task curriculum (teacher = {teacher_model_name})")
    _eprint("=" * 60)
    _eprint(f"🤖 Loading teacher {teacher_model_name}...")
    teacher, teacher_tok = llm.load_teacher(teacher_model_name)
    _eprint("✅ Teacher loaded\n")

    curriculum_data = curriculum.generate_curriculum(
        teacher, teacher_tok, structured, questions, domain
    )
    _eprint(f"✅ {len(curriculum_data)} curriculum examples (8 tasks)\n")
    with open(output_dir / "curriculum_data.json", "w") as f:
        json.dump(curriculum_data, f, indent=2)

    # ── 6. Surface filter (refusals/placeholders) + grounding validation ──
    _eprint("=" * 60)
    _eprint("STEP 6: 🧹 surface filter + grounding validation (32B judge)")
    _eprint("=" * 60)

    surface_clean = qa.filter_clean(curriculum_data)
    _eprint(f"  surface filter: {len(surface_clean)} kept (dropped {len(curriculum_data) - len(surface_clean)})")

    grounded, dropped_hallu = qa.validate_grounded(
        teacher, teacher_tok, surface_clean
    )
    _eprint(f"✅ {len(grounded)} grounded examples after validation\n")

    # ── 6.5. Multi-pass refinement (critique → rewrite) ────────────────
    _eprint("=" * 60)
    _eprint("STEP 6.5: ✨ multi-pass refinement (critique → rewrite)")
    _eprint("=" * 60)
    refined = refine.refine_pairs(teacher, teacher_tok, grounded)
    n_refined = sum(1 for p in refined if p.get("refined"))
    _eprint(f"✅ {len(refined)} refined ({n_refined} rewritten, "
            f"{len(refined) - n_refined} kept as-is)\n")

    # ── 6.7. Quality classifier (0-5 scoring, drop below 3) ────────────
    _eprint("=" * 60)
    _eprint("STEP 6.7: ⭐ quality scoring (0-5)")
    _eprint("=" * 60)
    scored = quality.score_quality(teacher, teacher_tok, refined)
    high_quality, score_dist = quality.filter_by_quality(scored, min_score=3)
    _eprint(f"✅ {len(high_quality)} high-quality pairs (≥3/5)\n")

    # ── 6.8. DPO data generation (preference pairs) ────────────────────
    _eprint("=" * 60)
    _eprint("STEP 6.8: 🎯 DPO preference pair generation")
    _eprint("=" * 60)
    dpo_triples = dpo.generate_rejected_answers(teacher, teacher_tok, high_quality)
    _eprint(f"✅ {len(dpo_triples)} DPO preference triples\n")

    with open(output_dir / "training_data_v2.json", "w") as f:
        json.dump(curriculum_data, f, indent=2)
    with open(output_dir / "training_data_clean.json", "w") as f:
        json.dump(high_quality, f, indent=2)
    with open(output_dir / "training_data_dpo.json", "w") as f:
        json.dump(dpo_triples, f, indent=2)

    # ── Free 32B before training (run_training reloads 7B) ───────────
    _eprint("🧹 Freeing 32B teacher from GPU...")
    llm.unload_model(teacher)
    del teacher_tok
    _eprint("✅ Memory cleared\n")

    elapsed = time.time() - started
    metadata = {
        "domain": domain,
        "num_input_docs": len(docs),
        "dropped_duplicates": n_dropped_dups,
        "pii_redacted_docs": n_pii_redacted,
        "schema_fields": len(sch),
        "lookup_fields": lookup_fields,
        "reasoning_fields": reasoning_fields,
        "structured_count": len(structured),
        "extraction_failed": failed,
        "curriculum_pairs": len(curriculum_data),
        "training_pairs": len(curriculum_data),  # alias for back-compat
        "after_surface_filter": len(surface_clean),
        "dropped_hallucination": dropped_hallu,
        "after_grounding": len(grounded),
        "refined_pairs": n_refined,
        "quality_score_distribution": score_dist,
        "high_quality_pairs": len(high_quality),
        "clean_pairs": len(high_quality),  # final training set
        "dpo_triples": len(dpo_triples),
        "teacher_model": teacher_model_name,
        "data_prep_seconds": round(elapsed, 1),
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    _eprint(f"⏱️  data prep took {elapsed/60:.1f} min")
    return metadata


def run_training_llamafactory(output_dir: Path, config_path: Path) -> dict:
    """Phase 9 alternative: train via LLaMA-Factory CLI.

    Replaces our hand-rolled trl/peft loop with LLaMA-Factory's optimized
    training. Free wins: FlashAttention-2, NEFTune, better LR scheduling,
    quantized training paths, less code to maintain.

    Process:
      1. Read training_data_clean.json from output_dir
      2. Register dataset under LLaMA-Factory's data/dataset_info.json
      3. Override output_dir in config to point at adapter/
      4. subprocess.run llamafactory-cli train

    The config (configs/lora.yaml) is the source of truth for all training
    hyperparameters. We only override output paths.
    """
    import subprocess
    import shutil

    clean_path = output_dir / "training_data_clean.json"
    if not clean_path.exists():
        raise SystemExit(f"no training data at {clean_path}")

    # Ensure llamafactory-cli is on PATH
    if shutil.which("llamafactory-cli") is None:
        raise SystemExit(
            "llamafactory-cli not found. Install: pip install llamafactory"
        )

    # 1. Register dataset with LLaMA-Factory.
    # Its data/dataset_info.json uses {dataset_name: {file_name, columns, ...}}.
    # We point it at our training_data_clean.json (alpaca format already).
    lf_data_dir = output_dir / "lf_data"
    lf_data_dir.mkdir(exist_ok=True)
    shutil.copy(clean_path, lf_data_dir / "aasaan_curriculum.json")
    dataset_info = {
        "aasaan_curriculum": {
            "file_name": "aasaan_curriculum.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
    }
    (lf_data_dir / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2))

    # 2. Build CLI command with output_dir + dataset_dir overrides.
    adapter_dir = output_dir / "adapter"
    cmd = [
        "llamafactory-cli", "train", str(config_path),
        f"output_dir={adapter_dir}",
        f"dataset_dir={lf_data_dir}",
    ]
    _eprint(f"🚀 Running: {' '.join(cmd)}")

    started = time.time()
    proc = subprocess.run(cmd, check=False)
    elapsed = time.time() - started

    if proc.returncode != 0:
        raise SystemExit(f"llamafactory-cli failed with exit code {proc.returncode}")

    _eprint(f"✅ LLaMA-Factory training done in {elapsed/60:.1f} min")
    _eprint(f"📁 Adapter at {adapter_dir}")

    # Update metadata
    meta_path = output_dir / "run_metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    metadata["training_seconds"] = round(elapsed, 1)
    metadata["training_backend"] = "llamafactory"
    metadata["training_config"] = str(config_path)
    meta_path.write_text(json.dumps(metadata, indent=2))
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
    parser.add_argument(
        "--use-llamafactory",
        action="store_true",
        help="train via LLaMA-Factory CLI (uses configs/lora.yaml). "
             "Adds FlashAttention-2, NEFTune, GaLore optimizations.",
    )
    parser.add_argument(
        "--lf-config",
        type=Path,
        default=Path("configs/lora.yaml"),
        help="LLaMA-Factory training config (only used with --use-llamafactory)",
    )

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
        if args.use_llamafactory:
            run_training_llamafactory(args.output, args.lf_config)
        else:
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
        if args.use_llamafactory:
            run_training_llamafactory(args.output, args.lf_config)
        else:
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
