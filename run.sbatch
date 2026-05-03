#!/bin/bash
#SBATCH --job-name=aasaan-pipeline
#SBATCH --account=dferris
#SBATCH --qos=dferris
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --output=logs/aasaan-%j.out

# Usage:
#   sbatch run.sbatch <input-file> [<column>] [<num-docs>]
#
# Examples:
#   sbatch run.sbatch /blue/dferris/y.sarwaiya/test/mtsample/mtsamples.csv
#   sbatch run.sbatch ./contracts.pdf
#   sbatch run.sbatch ./crm_notes.csv content 500

set -euo pipefail

mkdir -p logs

module load conda
conda activate doctorai

export HF_HOME=/blue/dferris/y.sarwaiya/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME

INPUT="${1:?usage: sbatch run.sbatch <input-file> [<column>] [<num-docs>]}"
COLUMN="${2:-}"
NUM="${3:-250}"

OUTDIR="./output_$(date +%Y%m%d_%H%M%S)"

ARGS=(--input "$INPUT" --num "$NUM" --output "$OUTDIR" --train --v3-mode)
if [[ -n "$COLUMN" ]]; then
  ARGS+=(--column "$COLUMN")
fi

echo "[$(date)] starting aasaan pipeline → $OUTDIR"
python run.py "${ARGS[@]}"
echo "[$(date)] done → $OUTDIR"
