#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/root/autodl-tmp/merged_models/law_qa_dpo"
PORT=6006
MODEL_NAME="law-expert"

if [ ! -d "$MODEL_PATH" ]; then
  echo "Model path not found: $MODEL_PATH"
  exit 1
fi

vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port $PORT \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching