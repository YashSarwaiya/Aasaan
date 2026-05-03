"""
Three-way side-by-side evaluation:

  1. baseline    stock Llama 3.1 8B, given the raw note only
  2. trained     base + your LoRA adapter, given the raw note only
  3. rag         stock Llama given the EXTRACTED STRUCTURED FORM (as JSON) instead of raw note

Same questions, same held-out test notes for all three.

Test notes are sampled from the same CSV but with a DIFFERENT random seed than
training, so we score on docs the model has not seen.

Usage:
    python evaluate.py \\
        --adapter ./output_20260426_213903/adapter \\
        --structured ./output_20260426_213903/structured.json \\
        --csv mtsample/mtsamples.csv \\
        --column transcription \\
        --num-test 10

After it finishes you get:
  eval_results.json   per-question generations from all three modes
  eval_summary.txt    accuracy table + side-by-side preview

Then look at eval_results.json with your eyes — it's faster than an LLM judge
for ~40 answers and you'll spot patterns the judge would miss.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
QUESTIONS = [
    "What is the diagnosis?",
    "What medications is the patient on?",
    "What is the plan?",
    "What is the patient's age?",
]


def generate(model, tok, prompt: str, max_tokens: int = 200) -> str:
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=3000).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            repetition_penalty=1.15,
        )
    text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return text.split("###")[0].strip()


def prompt_raw(question: str, note: str) -> str:
    return f"### Instruction: {question}\n### Input: {note[:2500]}\n### Output:"


def prompt_rag(question: str, structured: dict) -> str:
    ctx = json.dumps(structured, indent=2)[:2000]
    return (
        f"### Instruction: {question}\n"
        f"### Input: Structured form for this case:\n{ctx}\n"
        f"### Output:"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True, help="path to trained LoRA adapter")
    parser.add_argument("--schema", type=Path, required=True, help="path to schema.json (saved by run.py)")
    parser.add_argument("--csv", type=Path, required=True, help="source CSV (same one used for training)")
    parser.add_argument("--column", default="transcription")
    parser.add_argument("--num-test", type=int, default=10)
    parser.add_argument("--seed", type=int, default=999, help="different from training seed (42)")
    parser.add_argument("--output", type=Path, default=Path("./eval_results.json"))
    args = parser.parse_args()

    if "HF_HOME" not in os.environ:
        candidate = "/blue/dferris/y.sarwaiya/hf_cache"
        if os.path.isdir(os.path.dirname(candidate)):
            os.environ["HF_HOME"] = candidate

    # 1. Pick test notes (different seed → different docs than training)
    df = pd.read_csv(args.csv).dropna(subset=[args.column])
    test_df = df.sample(n=args.num_test, random_state=args.seed).reset_index(drop=True)
    test_notes = test_df[args.column].tolist()
    print(f"📂 picked {len(test_notes)} test notes (seed={args.seed})")

    # 2. Load schema (we'll extract structured forms for the test notes on the fly)
    with open(args.schema) as f:
        schema_payload = json.load(f)
    schema = schema_payload["schema"]
    domain = schema_payload.get("domain", "your domain")

    # 3. Load base + tokenizer
    print(f"🤖 loading {BASE_MODEL}...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="auto")
    base.eval()
    print("✅ base model loaded")

    # 4. Extract structured forms for the held-out test notes (used by RAG mode).
    #    Same logic the pipeline used during training, just on different docs.
    print("\n" + "=" * 60)
    print("EXTRACTING structured forms for the 10 held-out test notes")
    print("=" * 60)
    from pipeline.extract import batch_extract_structured

    structured_results, _ = batch_extract_structured(
        base, tok, test_notes, domain, schema, batch_size=4,
    )
    structured_by_idx: dict[int, dict] = {item["id"]: item["structured"] for item in structured_results}
    print(f"✅ {len(structured_by_idx)}/{len(test_notes)} extracted")

    def find_structured(note_idx: int) -> dict | None:
        return structured_by_idx.get(note_idx)

    # ── Mode 1: baseline (raw note → stock Llama) ────────────────────────
    print("\n" + "=" * 60)
    print("MODE 1: baseline (raw note, no training)")
    print("=" * 60)
    baseline_answers = []
    for i, note in enumerate(test_notes):
        for q in QUESTIONS:
            ans = generate(base, tok, prompt_raw(q, note))
            baseline_answers.append({"note_idx": i, "question": q, "answer": ans})
        print(f"  {i + 1}/{len(test_notes)}")

    # ── Mode 3: RAG (stock Llama + structured form as context) ───────────
    #    Note: we keep the same base model loaded, no PEFT yet
    print("\n" + "=" * 60)
    print("MODE 3: RAG (stock Llama + structured form)")
    print("=" * 60)
    rag_answers = []
    for i, note in enumerate(test_notes):
        structured = find_structured(i)
        for q in QUESTIONS:
            if structured is None:
                ans = "[skipped — extraction failed for this test note]"
            else:
                ans = generate(base, tok, prompt_rag(q, structured))
            rag_answers.append({"note_idx": i, "question": q, "answer": ans, "had_form": structured is not None})
        print(f"  {i + 1}/{len(test_notes)}")

    # ── Mode 2: trained adapter ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODE 2: trained (base + LoRA adapter)")
    print("=" * 60)
    print("🔄 loading LoRA adapter...")
    trained = PeftModel.from_pretrained(base, str(args.adapter))
    trained.eval()
    print("✅ adapter loaded")

    trained_answers = []
    for i, note in enumerate(test_notes):
        for q in QUESTIONS:
            ans = generate(trained, tok, prompt_raw(q, note))
            trained_answers.append({"note_idx": i, "question": q, "answer": ans})
        print(f"  {i + 1}/{len(test_notes)}")

    # ── Mode 4: trained + RAG (the real hybrid) ──────────────────────────
    # Builds an in-memory RAG index over the structured forms of test notes
    # (just the lookup-style fields) and queries with the trained model.
    # This is the mode that should win the hybrid claim.
    print("\n" + "=" * 60)
    print("MODE 4: trained + RAG (LoRA + retrieval over structured forms)")
    print("=" * 60)
    try:
        from pipeline.rag import build_rag_index, rag_answer

        rag_items = [
            {"id": item["id"], "structured": item["structured"]}
            for item in structured_results
        ]
        # Index lookup-style fields if schema_payload provides them, else all
        index_fields = schema_payload.get("lookup_fields") or None
        rag_handle = build_rag_index(rag_items, fields=index_fields)
        print(f"✅ RAG index: {rag_handle.get('n_chunks', 0)} chunks "
              f"(backend={rag_handle.get('_backend')})")

        trained_rag_answers = []
        for i, note in enumerate(test_notes):
            for q in QUESTIONS:
                ans = rag_answer(rag_handle, q, trained, tok, top_k=5)
                trained_rag_answers.append({
                    "note_idx": i, "question": q, "answer": ans
                })
            print(f"  {i + 1}/{len(test_notes)}")
    except Exception as e:
        print(f"⚠️ trained+RAG mode failed: {e}")
        trained_rag_answers = []

    # ── Save ──────────────────────────────────────────────────────────────
    results = {
        "test_notes": [{"idx": i, "preview": n[:300]} for i, n in enumerate(test_notes)],
        "questions": QUESTIONS,
        "baseline": baseline_answers,
        "rag": rag_answers,
        "trained": trained_answers,
        "trained_rag": trained_rag_answers,
    }
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\n📁 wrote {args.output}")

    # ── Side-by-side preview ──────────────────────────────────────────────
    summary = ["=" * 60, "SIDE-BY-SIDE PREVIEW (first 3 test cases)", "=" * 60, ""]
    for i in range(min(3, len(test_notes))):
        summary.append(f"--- Test note {i + 1} ---")
        summary.append(f"NOTE: {test_notes[i][:300]}...\n")
        for q in QUESTIONS:
            base_ans = next(a["answer"] for a in baseline_answers if a["note_idx"] == i and a["question"] == q)
            train_ans = next(a["answer"] for a in trained_answers if a["note_idx"] == i and a["question"] == q)
            rag_ans = next(a["answer"] for a in rag_answers if a["note_idx"] == i and a["question"] == q)
            summary.append(f"Q: {q}")
            summary.append(f"  baseline: {base_ans[:200]}")
            summary.append(f"  trained : {train_ans[:200]}")
            summary.append(f"  rag     : {rag_ans[:200]}")
            summary.append("")
        summary.append("")

    Path("./eval_summary.txt").write_text("\n".join(summary))
    print("📁 wrote eval_summary.txt")
    print("\n" + "\n".join(summary[:30]))


if __name__ == "__main__":
    main()
