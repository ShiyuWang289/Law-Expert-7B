#!/usr/bin/env bash
set -euo pipefail
TAG=${1:-fp16_baseline}
MODEL="law-expert-fp16"            # ← 与 vLLM 的 --served-model-name 一致

python bench/bench_vllm_openai.py --model "$MODEL" --concurrency 1  --requests 20 --max_tokens 128 --tag "$TAG"
python bench/bench_vllm_openai.py --model "$MODEL" --concurrency 4  --requests 20 --max_tokens 128 --tag "$TAG"
python bench/bench_vllm_openai.py --model "$MODEL" --concurrency 8  --requests 20 --max_tokens 128 --tag "$TAG"
python bench/bench_vllm_openai.py --model "$MODEL" --concurrency 16 --requests 40 --max_tokens 128 --tag "$TAG"