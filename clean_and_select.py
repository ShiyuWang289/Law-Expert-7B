# clean_and_select.py
import json
import random

input_file = "./raw_data/DISC-Law-SFT/DISC-Law-SFT-Pair-QA-released.jsonl"

# ========== 第一步：读取全部数据 ==========
all_data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        if 'input' in data and 'output' in data:
            all_data.append(data)

print(f"原始数据总量: {len(all_data)} 条")

# ========== 第二步：数据清洗 ==========
cleaned = []
seen = set()  # 用于去重

for item in all_data:
    inp = item['input'].strip()
    out = item['output'].strip()
    
    # 规则 1：去空值
    if not inp or not out:
        continue
    
    # 规则 2：去重（基于 input 去重）
    if inp in seen:
        continue
    seen.add(inp)
    
    # 规则 3：过滤过短的回答（低质量）
    if len(out) < 30:
        continue
    
    # 规则 4：过滤过长的样本（避免超出上下文长度）
    if len(inp) + len(out) > 2000:
        continue
    
    # 规则 5：过滤明显的噪声（可根据实际情况添加）
    if "http" in out or "www." in out:
        continue
    
    cleaned.append(item)

print(f"清洗后数据量: {len(cleaned)} 条")

# ========== 第三步：随机采样 500 条 ==========
random.seed(42)  # 固定随机种子，保证可复现
selected = random.sample(cleaned, min(500, len(cleaned)))
print(f"最终选取: {len(selected)} 条")

# ========== 第四步：保存清洗后的数据 ==========
with open("./raw_data/law_qa_cleaned.jsonl", 'w', encoding='utf-8') as f:
    for item in selected:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print("✅ 清洗完成，保存至 ./raw_data/law_qa_cleaned.jsonl")

# ========== 统计信息 ==========
input_lens = [len(item['input']) for item in selected]
output_lens = [len(item['output']) for item in selected]
print(f"\n📊 数据统计:")
print(f"  input  平均长度: {sum(input_lens)/len(input_lens):.0f} 字符")
print(f"  output 平均长度: {sum(output_lens)/len(output_lens):.0f} 字符")
print(f"  input  最大长度: {max(input_lens)} 字符")
print(f"  output 最大长度: {max(output_lens)} 字符")