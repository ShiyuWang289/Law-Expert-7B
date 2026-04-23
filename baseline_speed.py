# baseline_speed.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, time, json

model_path = "/root/autodl-tmp/merged_models/law_qa_dpo"
TEST_QUESTIONS = [
    "劳动合同到期公司不续签，员工能获得赔偿吗？",
    "交通事故责任认定不服怎么办？",
    "签了竞业协议离职后公司不给补偿金怎么办？",
]
SYSTEM = "你是一个专业的中国法律顾问，请给出准确、专业的回答。"

print("加载 bf16 模型...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
print("✅ 加载完成\n")

results = []
total_tokens, total_time = 0, 0

for q in TEST_QUESTIONS:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=256,
            temperature=0.1, do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = time.time() - t0

    output_ids = outputs[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    tps = len(output_ids) / elapsed

    print(f"问题: {q}")
    print(f"回答: {answer[:200]}{'...' if len(answer)>200 else ''}")
    print(f"速度: {tps:.1f} tokens/s ({len(output_ids)} tokens, {elapsed:.1f}s)\n")

    results.append({"question": q, "answer": answer, "tokens": len(output_ids),
                    "time": elapsed, "tps": tps})
    total_tokens += len(output_ids)
    total_time += elapsed

avg_tps = total_tokens / total_time
print("="*50)
print(f"📊 bf16 平均速度: {avg_tps:.1f} tokens/s")
print("="*50)

# 保存基线数据供后续对比
with open("eval_results/baseline_speed.json", "w") as f:
    json.dump({"avg_tps": avg_tps, "results": results}, f, ensure_ascii=False, indent=2)
print("基线数据已保存 → eval_results/baseline_speed.json")