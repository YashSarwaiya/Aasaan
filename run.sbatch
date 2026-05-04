#!/bin/bash
#SBATCH --job-name=aasaan-pipeline
#SBATCH --account=dferris
#SBATCH --qos=dferris
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=06:00:00
#SBATCH --output=logs/aasaan-%j.out

# Usage:
#   sbatch run.sbatch <input-file> [<column>] [<num-docs>] [<version>]
#
# Versions:
#   auto       — detect task shape after extraction, route automatically
#                (classification → DPO, extraction → refine, mixed → both).
#                DEFAULT. Recommended for non-technical users.
#   v3-refine  — refine alone. Best for extraction tasks (logs, forms,
#                structured records — verbatim outputs).
#   v3-dpo     — DPO alone. Best for classification tasks (clause types,
#                intents, categories — short label outputs).
#   v3-best    — refine + DPO together. Candidate universal default; use
#                when you don't trust the auto-detect (mixed/unusual data).
#
# Examples:
#   sbatch run.sbatch ./loghub_data/loghub_train.csv raw_log 200 auto
#   sbatch run.sbatch ./loghub_data/loghub_train.csv raw_log 200 v3-refine
#   sbatch run.sbatch ./cuad_data/cuad_train.csv clause_text 200 v3-dpo

set -euo pipefail

mkdir -p logs

module load conda
conda activate doctorai

export HF_HOME=/blue/dferris/y.sarwaiya/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME

INPUT="${1:?usage: sbatch run.sbatch <input-file> [<column>] [<num-docs>] [<version>]}"
COLUMN="${2:-}"
NUM="${3:-250}"
VERSION="${4:-auto}"

case "$VERSION" in
  auto)       EXTRA_FLAGS=(--auto) ;;
  v3-refine)  EXTRA_FLAGS=(--skip-dpo) ;;
  v3-dpo)     EXTRA_FLAGS=(--skip-refine) ;;
  v3-best)    EXTRA_FLAGS=() ;;  # refine + DPO both on
  *)          echo "unknown version: $VERSION (expected auto|v3-refine|v3-dpo|v3-best)" >&2; exit 1 ;;
esac

# LogHub (and any other short/templated input) benefits from disabling
# MinHash dedup, which would otherwise drop ~80% as near-duplicates.
if [[ "$INPUT" == *loghub* || "$INPUT" == *log* ]]; then
  EXTRA_FLAGS+=(--no-dedup)
fi

OUTDIR="./output_${VERSION}_$(date +%Y%m%d_%H%M%S)"

ARGS=(--input "$INPUT" --num "$NUM" --output "$OUTDIR" --train "${EXTRA_FLAGS[@]}")
if [[ -n "$COLUMN" ]]; then
  ARGS+=(--column "$COLUMN")
fi

echo "[$(date)] starting aasaan pipeline ($VERSION) → $OUTDIR"
python run.py "${ARGS[@]}"
echo "[$(date)] done → $OUTDIR"
