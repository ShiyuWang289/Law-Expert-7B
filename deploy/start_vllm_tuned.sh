#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/root/autodl-tmp/merged_models/law_qa_dpo"
PORT=6006
NAME="law-expert"

# 可改参数
GMU="${GMU:-0.90}"
MAX_LEN="${MAX_LEN:-2048}"
MAX_SEQS="${MAX_SEQS:-16}"   # 核心调优参数

vllm serve "$MODEL_PATH" \
  --served-model-name "$NAME" \
  --host 0.0.0.0 \
  --port $PORT \
  --dtype float16 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GMU" \
  --max-num-seqs "$MAX_SEQS" \
  --enable-prefix-caching