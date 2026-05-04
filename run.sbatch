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
#   v3         — lean baseline (no refine, no quality, no DPO)
#   v3-refine  — v3 + refine          (= v4)
#   v3-quality — v3 + quality filter  (ablation: does quality filter help?)
#   v3-dpo     — v3 + DPO             (ablation: does DPO help?)
#   v4         — v3 + refine          (alias for v3-refine)
#   v5         — everything on        (refine + quality + new_tasks + DPO)
#   v6         — refine + new_tasks   (no quality, no DPO)
#
# Examples:
#   sbatch run.sbatch ./loghub_data/loghub_train.csv raw_log 200 v3
#   sbatch run.sbatch ./loghub_data/loghub_train.csv raw_log 200 v3-quality
#   sbatch run.sbatch ./loghub_data/loghub_train.csv raw_log 200 v3-dpo

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
VERSION="${4:-v4}"

case "$VERSION" in
  v3)         EXTRA_FLAGS=(--v3-mode) ;;
  v3-refine)  EXTRA_FLAGS=(--skip-quality-filter --skip-new-tasks --skip-dpo) ;;
  v3-quality) EXTRA_FLAGS=(--skip-refine --skip-new-tasks --skip-dpo) ;;
  v3-dpo)     EXTRA_FLAGS=(--skip-refine --skip-quality-filter --skip-new-tasks) ;;
  v4)         EXTRA_FLAGS=(--skip-quality-filter --skip-new-tasks --skip-dpo) ;;
  v5)         EXTRA_FLAGS=() ;;
  v6)         EXTRA_FLAGS=(--skip-quality-filter --skip-dpo) ;;
  *)          echo "unknown version: $VERSION (expected v3|v3-refine|v3-quality|v3-dpo|v4|v5|v6)" >&2; exit 1 ;;
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
