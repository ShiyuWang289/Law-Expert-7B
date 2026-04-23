# convert_data.py
import json
import os

input_file = "./raw_data/law_qa_cleaned.jsonl"
SYSTEM_PROMPT = "你是一个专业的中国法律顾问，请根据用户的法律问题，给出准确、专业、有法律依据的回答。"
INSTRUCTION = "你是一个法律专家，请根据用户的问题给出专业的回答"

# 读取清洗后的数据
raw_data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        raw_data.append(json.loads(line.strip()))

print(f"读取 {len(raw_data)} 条数据")

# ========================================
# 转换为 Alpaca 格式
# ========================================
alpaca_data = []
for item in raw_data:
    alpaca_data.append({
        "instruction": INSTRUCTION,
        "input": item["input"],
        "output": item["output"],
        "system": SYSTEM_PROMPT
    })

# 保存到 LLaMA-Factory/data/ 目录
os.makedirs("data", exist_ok=True)
alpaca_output = "data/law_qa_alpaca.json"
with open(alpaca_output, 'w', encoding='utf-8') as f:
    json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

print(f"✅ Alpaca 格式已保存: {alpaca_output} ({len(alpaca_data)} 条)")

# ========================================
# 转换为 ShareGPT 格式
# ========================================
sharegpt_data = []
for item in raw_data:
    sharegpt_data.append({
        "conversations": [
            {"from": "human", "value": item["input"]},
            {"from": "gpt", "value": item["output"]}
        ],
        "system": SYSTEM_PROMPT
    })

sharegpt_output = "data/law_qa_sharegpt.json"
with open(sharegpt_output, 'w', encoding='utf-8') as f:
    json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)

print(f"✅ ShareGPT 格式已保存: {sharegpt_output} ({len(sharegpt_data)} 条)")

# ========================================
# 验证：打印各一条样例
# ========================================
print("\n" + "="*60)
print("📋 Alpaca 格式样例：")
print(json.dumps(alpaca_data[0], ensure_ascii=False, indent=2)[:500])
print("\n" + "="*60)
print("📋 ShareGPT 格式样例：")
print(json.dumps(sharegpt_data[0], ensure_ascii=False, indent=2)[:500])