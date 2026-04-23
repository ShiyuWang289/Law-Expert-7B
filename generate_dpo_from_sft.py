# generate_dpo_from_sft.py
import json
import random

# 读取 SFT 数据
with open("data/law_qa_alpaca.json", 'r', encoding='utf-8') as f:
    sft_data = json.load(f)

dpo_data = []
random.seed(42)

# 从 SFT 数据中采样，SFT 的 output 作为 chosen，构造简化版作为 rejected
for item in random.sample(sft_data, min(50, len(sft_data))):
    original_output = item["output"]
    
    # 构造 rejected：截取前30%内容 + 加一句模糊的结尾
    truncated = original_output[:int(len(original_output) * 0.3)]
    rejected = truncated + "建议您咨询专业律师获取更详细的法律建议。"
    
    dpo_data.append({
        "instruction": item.get("instruction", "你是一个法律专家，请根据用户的问题给出专业的回答"),
        "input": item["input"],
        "chosen": original_output,
        "rejected": rejected
    })

# 合并手工数据
with open("data/law_qa_dpo.json", 'r', encoding='utf-8') as f:
    manual_data = json.load(f)

all_dpo_data = manual_data + dpo_data
print(f"手工数据: {len(manual_data)} 条")
print(f"自动构造: {len(dpo_data)} 条")
print(f"合并总计: {len(all_dpo_data)} 条")

with open("data/law_qa_dpo.json", 'w', encoding='utf-8') as f:
    json.dump(all_dpo_data, f, ensure_ascii=False, indent=2)

print("✅ 合并后的 DPO 数据集已保存")