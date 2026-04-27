#!/bin/bash
# Live-output run: requests a B200 interactively and runs run.py with stdout
# streamed straight to your terminal. No SLURM log file required.
#
# Usage:
#   ./run-interactive.sh <input-file> [<column>] [<num-docs>]
#
# Examples:
#   ./run-interactive.sh /blue/dferris/y.sarwaiya/test/mtsample/mtsamples.csv
#   ./run-interactive.sh ./contracts.pdf
#   ./run-interactive.sh ./crm_notes.csv content 500

set -euo pipefail

INPUT="${1:?usage: ./run-interactive.sh <input-file> [<column>] [<num-docs>]}"
COLUMN="${2:-}"
NUM="${3:-250}"
OUTDIR="./output_$(date +%Y%m%d_%H%M%S)"

ARGS=(--input "$INPUT" --num "$NUM" --output "$OUTDIR" --train)
if [[ -n "$COLUMN" ]]; then
  ARGS+=(--column "$COLUMN")
fi

echo "[$(date)] requesting B200 interactive session..."
srun --account=dferris --qos=dferris --partition=hpg-b200 \
     --gres=gpu:1 --cpus-per-task=4 --mem=64gb --time=04:00:00 \
     --pty bash -c "
module load conda
conda activate doctorai
export HF_HOME=/blue/dferris/y.sarwaiya/hf_cache
export TRANSFORMERS_CACHE=\$HF_HOME
export HF_HUB_CACHE=\$HF_HOME
echo '[\$(date)] running pipeline → ${OUTDIR}'
python run.py ${ARGS[@]}
echo '[\$(date)] done → ${OUTDIR}'
"
