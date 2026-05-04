#!/bin/bash
#SBATCH --job-name=aasaan-direct-sft
#SBATCH --account=dferris
#SBATCH --qos=dferris
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/aasaan-%j.out

# Usage:
#   sbatch train_direct.sbatch <input-csv> <input-col> <output-col> <instruction> <num>
#
# Example (CUAD direct SFT, 200 rows):
#   sbatch train_direct.sbatch \
#       ./cuad_data/cuad_train.csv \
#       clause_text \
#       label \
#       "Classify this contract clause." \
#       200

set -euo pipefail
mkdir -p logs

module load conda
conda activate doctorai

export HF_HOME=/blue/dferris/y.sarwaiya/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME

INPUT="${1:?usage: sbatch train_direct.sbatch <csv> <input-col> <output-col> <instruction> <num>}"
INPUT_COL="${2:?missing input-col}"
OUTPUT_COL="${3:?missing output-col}"
INSTRUCTION="${4:?missing instruction}"
NUM="${5:-200}"

OUTDIR="./output_direct_$(date +%Y%m%d_%H%M%S)"

echo "[$(date)] starting direct SFT (NO pipeline) → $OUTDIR"
python train_direct.py \
  --input "$INPUT" \
  --input-col "$INPUT_COL" \
  --output-col "$OUTPUT_COL" \
  --instruction "$INSTRUCTION" \
  --num "$NUM" \
  --output "$OUTDIR"
echo "[$(date)] done → $OUTDIR"
