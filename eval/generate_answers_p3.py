#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

QUESTIONS = "eval/p3_eval_questions.json"
OUT_A = "eval/answers_A.json"
OUT_B = "eval/answers_B.json"

MODEL_A = "/root/autodl-tmp/merged_models/law_qa_p3_A"
MODEL_B = "/root/autodl-tmp/merged_models/law_qa_p3_B"

SYSTEM_PROMPT = "你是一个专业的中国法律顾问，请根据用户的法律问题，给出准确、专业、有法律依据的回答。"

GEN_CFG = dict(
    max_new_tokens=512,
    do_sample=False,   # 为可复现，关闭采样
    temperature=1.0,
    top_p=1.0
)

def load_model(model_path):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    mdl.eval()
    return tok, mdl

def one_answer(tok, mdl, q):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q}
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(mdl.device)

    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            **GEN_CFG
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    ans = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return ans

def run(model_path, out_path):
    tok, mdl = load_model(model_path)
    qs = json.load(open(QUESTIONS, "r", encoding="utf-8"))
    res = []
    for x in qs:
        ans = one_answer(tok, mdl, x["question"])
        res.append({"id": x["id"], "answer": ans})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"✅ saved: {out_path}")

if __name__ == "__main__":
    run(MODEL_A, OUT_A)
    run(MODEL_B, OUT_B)