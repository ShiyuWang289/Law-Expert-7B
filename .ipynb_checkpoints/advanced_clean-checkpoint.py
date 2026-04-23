# advanced_clean.py
import json
import re
from collections import Counter

input_file = "data/law_qa_alpaca.json"

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"清洗前: {len(data)} 条")

# ========================================
# 维度 1：基础清洗（Step 1 已做，这里补充检查）
# ========================================
basic_cleaned = []
duplicate_inputs = set()

for item in data:
    inp = item.get("input", "").strip()
    out = item.get("output", "").strip()
    
    # 去空值
    if not inp or not out:
        continue
    
    # 去重（基于 input）
    if inp in duplicate_inputs:
        continue
    duplicate_inputs.add(inp)
    
    basic_cleaned.append(item)

print(f"基础清洗后: {len(basic_cleaned)} 条")

# ========================================
# 维度 2：质量过滤
# ========================================
quality_cleaned = []
reject_reasons = Counter()

for item in basic_cleaned:
    inp = item["input"].strip()
    out = item["output"].strip()
    
    # --- 2a. 长度过滤 ---
    # 问题太短（可能没有足够信息）
    if len(inp) < 5:
        reject_reasons["input_too_short"] += 1
        continue
    
    # 回答太短（可能是低质量回复）
    if len(out) < 30:
        reject_reasons["output_too_short"] += 1
        continue
    
    # 总长度控制（避免超出模型 cutoff_len）
    # Qwen2.5 支持 128K，但训练时 cutoff_len 通常设 1024~2048
    if len(inp) + len(out) > 1500:
        reject_reasons["total_too_long"] += 1
        continue
    
    # --- 2b. 噪声过滤 ---
    # 过滤包含 URL 的条目
    if re.search(r'https?://|www\.', out):
        reject_reasons["contains_url"] += 1
        continue
    
    # 过滤乱码/特殊字符过多的条目
    special_char_ratio = len(re.findall(r'[^\u4e00-\u9fff\w\s，。、；：""''！？（）《》\-]', out)) / max(len(out), 1)
    if special_char_ratio > 0.15:
        reject_reasons["too_many_special_chars"] += 1
        continue
    
    # 过滤纯复制问题的回答（鹦鹉回复）
    if out.startswith(inp[:20]):
        reject_reasons["parrot_response"] += 1
        continue
    
    # --- 2c. 法律领域相关性 ---
    # 简单启发式：回答中至少包含一些法律相关词汇
    legal_keywords = ["法", "条", "规定", "权", "合同", "责任", "赔偿", 
                      "诉讼", "法院", "律师", "当事人", "民事", "刑事",
                      "劳动", "婚姻", "继承", "侵权", "知识产权"]
    has_legal_content = any(kw in out for kw in legal_keywords)
    if not has_legal_content:
        reject_reasons["not_legal_related"] += 1
        continue
    
    quality_cleaned.append(item)

print(f"质量过滤后: {len(quality_cleaned)} 条")
print(f"\n📊 过滤原因统计:")
for reason, count in reject_reasons.most_common():
    print(f"   {reason}: {count} 条")

# ========================================
# 维度 3：多样性优化
# ========================================
# 简易语义去重：基于问题前 N 个字符的相似度
# （工业级方案会用 Embedding 向量相似度，这里用简化版）
diverse_cleaned = []
seen_prefixes = set()
PREFIX_LEN = 15  # 前15个字符相同视为近似重复

for item in quality_cleaned:
    prefix = item["input"].strip()[:PREFIX_LEN]
    if prefix in seen_prefixes:
        reject_reasons["semantic_duplicate"] += 1
        continue
    seen_prefixes.add(prefix)
    diverse_cleaned.append(item)

print(f"多样性优化后: {len(diverse_cleaned)} 条")

# ========================================
# 维度 4：最终统计与保存
# ========================================
final_data = diverse_cleaned

# 统计
input_lens = [len(item["input"]) for item in final_data]
output_lens = [len(item["output"]) for item in final_data]

print(f"\n{'='*50}")
print(f"📋 最终数据集统计:")
print(f"   总条数: {len(final_data)}")
print(f"   input  平均长度: {sum(input_lens)/len(input_lens):.0f} 字")
print(f"   output 平均长度: {sum(output_lens)/len(output_lens):.0f} 字")
print(f"   input  范围: [{min(input_lens)}, {max(input_lens)}] 字")
print(f"   output 范围: [{min(output_lens)}, {max(output_lens)}] 字")

# 保存最终版本
output_file = "data/law_qa_alpaca.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

print(f"\n✅ 最终数据已保存: {output_file}")