#!/bin/bash
# One-time setup on a CUDA GPU machine (e.g. a Vast.ai PyTorch instance with >= 40 GB disk).
# Installs the Python packages and pre-downloads only the weights the experiment needs.
set -euo pipefail
cd "$(dirname "$0")"
GENERATOR="${GENERATOR:-Qwen/Qwen2.5-7B-Instruct}"
pip install -q -r requirements.txt
for repo in "$GENERATOR" sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5; do
  hf download "$repo" --include "*.json" --include "*.safetensors" --include "*.txt" --include "*.model" \
    --include "1_Pooling/*" --include "*.jinja" > /dev/null
  echo "downloaded $repo"
done
python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
PY
df -h . | tail -1
