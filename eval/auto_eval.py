#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法：
python eval/auto_eval.py \
  --model_name base \
  --model_path /root/autodl-tmp/models/Qwen/Qwen2_5-7B-Instruct \
  --template qwen \
  --run_tag run_2026_04_26

python eval/auto_eval.py \
  --model_name sft \
  --model_path /root/autodl-tmp/merged_models/law_qa_sft \
  --template qwen \
  --run_tag run_2026_04_26
"""
import os, json, argparse, datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from metrics import score_case

SYSTEM_PROMPT = "你是一个专业的中国法律顾问，请根据用户的法律问题，给出准确、专业、有法律依据的回答。"

def gen_answer(model, tok, q):
    messages = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":q}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tok.eos_token_id
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, choices=["base","sft","dpo"])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--template", default="qwen")
    ap.add_argument("--golden", default="eval/golden_cases.json")
    ap.add_argument("--run_tag", default=datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    args = ap.parse_args()

    cases = json.load(open(args.golden, "r", encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    rows = []
    for c in cases:
        ans = gen_answer(model, tok, c["question"])
        m = score_case(ans, c)
        rows.append({
            "id": c["id"],
            "category": c["category"],
            "question": c["question"],
            "answer": ans,
            **m
        })

    out_dir = f"eval/runs/{args.run_tag}"
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"{out_dir}/{args.model_name}_results.json"
    json.dump(rows, open(out_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # summary
    n = len(rows)
    avg = lambda k: sum(x[k] for x in rows) / n
    summary = {
        "model": args.model_name,
        "model_path": args.model_path,
        "run_tag": args.run_tag,
        "n_cases": n,
        "law_accuracy": avg("law_accuracy"),
        "coverage": avg("coverage"),
        "repetition_rate": avg("repetition_rate"),
        "avg_len": avg("avg_len"),
        "hallucination_count": sum(x["hallucination_count"] for x in rows)
    }
    json.dump(summary, open(f"{out_dir}/{args.model_name}_summary.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved:", out_file)
    print("saved:", f"{out_dir}/{args.model_name}_summary.json")

if __name__ == "__main__":
    main()