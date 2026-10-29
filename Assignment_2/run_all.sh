#!/bin/bash
# Full Assignment 2 pipeline (run on the GPU instance). Each step logs to logs/<step>.log.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
python -u build_dataset.py    2>&1 | grep --line-buffered -v Warning > logs/build_dataset.log
python -u prune_finetune.py   2>&1 | grep --line-buffered -v Warning > logs/prune_finetune.log
python -u distill.py          2>&1 | grep --line-buffered -v Warning > logs/distill.log
python -u evaluate.py --prompt P-2 --epochs 5  2>&1 | grep --line-buffered -v Warning > logs/evaluate.log
echo PIPELINE_DONE >> logs/evaluate.log
