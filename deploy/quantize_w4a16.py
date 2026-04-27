#!/usr/bin/env python3
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_PATH = "/root/autodl-tmp/merged_models/law_qa_dpo"
OUT_PATH = "/root/autodl-tmp/quantized_models/law_qa_dpo_w4a16"

recipe = QuantizationModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"]
)

oneshot(
    model=MODEL_PATH,
    recipe=recipe,
    output_dir=OUT_PATH
)
print("Quantized model saved to:", OUT_PATH)