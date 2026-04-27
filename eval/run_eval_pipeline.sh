#!/usr/bin/env bash
set -euo pipefail
cd ~/autodl-tmp/LLaMA-Factory

RUN_TAG=${1:-run_$(date +%Y%m%d_%H%M%S)}

BASE_PATH="/root/autodl-tmp/LLaMA-Factory/models/Qwen/Qwen2.5-7B-Instruct"
SFT_PATH="/root/autodl-tmp/merged_models/law_qa_sft"
DPO_PATH="/root/autodl-tmp/merged_models/law_qa_dpo"

python eval/auto_eval.py --model_name base --model_path "$BASE_PATH" --run_tag "$RUN_TAG" --template qwen
python eval/auto_eval.py --model_name sft  --model_path "$SFT_PATH"  --run_tag "$RUN_TAG" --template qwen
python eval/auto_eval.py --model_name dpo  --model_path "$DPO_PATH"  --run_tag "$RUN_TAG" --template qwen

python eval/regression_check.py --run_tag "$RUN_TAG"

echo "✅ done. run_tag=$RUN_TAG"
echo "See: eval/runs/$RUN_TAG and eval/reports/"