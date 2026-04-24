# advanced_clean.py
import json
import re
import os
from collections import Counter

input_file = "data/law_qa_alpaca.json"
output_file = "raw_data/law_qa_cleaned.jsonl"  # 保持你原流程可继续被 convert_data.py 使用

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"清洗前: {len(data)} 条")

# 统计：阶段样本数
stage_stats = {
    "raw_total": len(data),
    "after_basic_clean": 0,
    "after_quality_clean": 0
}

# ========================================
# 维度 1：基础清洗（Step 1）
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

stage_stats["after_basic_clean"] = len(basic_cleaned)
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
    special_char_ratio = len(
        re.findall(r'[^\u4e00-\u9fff\w\s，。、；：""\'\'！？（）《》\-]', out)
    ) / max(len(out), 1)
    if special_char_ratio > 0.15:
        reject_reasons["too_many_special_chars"] += 1
        continue

    # 过滤纯复制问题的回答（鹦鹉回复）
    if out.startswith(inp[:20]):
        reject_reasons["parrot_response"] += 1
        continue

    # --- 2c. 法律领域相关性 ---
    # 简单启发式：回答中至少包含一些法律相关词汇
    legal_keywords = [
        "法", "条", "规定", "权", "合同", "责任", "赔偿",
        "劳动", "仲裁", "起诉", "诉讼", "婚姻", "遗嘱", "财产"
    ]
    if not any(kw in out for kw in legal_keywords):
        reject_reasons["low_legal_relevance"] += 1
        continue

    quality_cleaned.append(item)

stage_stats["after_quality_clean"] = len(quality_cleaned)
print(f"质量过滤后: {len(quality_cleaned)} 条")

# ========================================
# 保存清洗后的数据（jsonl）
# ========================================
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    for item in quality_cleaned:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 清洗后数据已保存到: {output_file}")

# ========================================
# 新增：输出规则命中统计（P2）
# ========================================
os.makedirs("analysis", exist_ok=True)

quality_input = stage_stats["after_basic_clean"]
quality_removed = quality_input - stage_stats["after_quality_clean"]
raw_total = stage_stats["raw_total"]

def pct(num, den):
    return round((num / den * 100), 2) if den else 0.0

# 规则定义（用于 markdown 展示）
rule_desc = {
    "input_too_short": "len(input) < 5",
    "output_too_short": "len(output) < 30",
    "total_too_long": "len(input) + len(output) > 1500",
    "contains_url": "output contains URL (http/https/www)",
    "too_many_special_chars": "special_char_ratio(output) > 0.15",
    "parrot_response": "output.startswith(input[:20])",
    "low_legal_relevance": "no legal keywords in output"
}

# JSON 结构化结果
rule_hits = {
    "input_file": input_file,
    "output_file": output_file,
    "stage_stats": {
        "raw_total": raw_total,
        "after_basic_clean": stage_stats["after_basic_clean"],
        "quality_stage_input": quality_input,
        "after_quality_clean": stage_stats["after_quality_clean"],
        "quality_stage_removed": quality_removed
    },
    "quality_rules": []
}

for k, v in reject_reasons.most_common():
    rule_hits["quality_rules"].append({
        "rule_key": k,
        "rule": rule_desc.get(k, k),
        "hits": v,
        "hit_ratio_in_quality_stage_pct": pct(v, quality_input),
        "contribution_in_raw_total_pct": pct(v, raw_total)
    })

json_out = "analysis/rule_hits.json"
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(rule_hits, f, ensure_ascii=False, indent=2)

# Markdown 报告
md_out = "analysis/rule_hits.md"
with open(md_out, "w", encoding="utf-8") as f:
    f.write("# 清洗规则命中统计（P2）\n\n")
    f.write(f"- 输入文件: `{input_file}`\n")
    f.write(f"- 输出文件: `{output_file}`\n\n")
    f.write("## 阶段统计\n\n")
    f.write("| 阶段 | 样本数 |\n")
    f.write("|---|---:|\n")
    f.write(f"| 原始样本 | {raw_total} |\n")
    f.write(f"| 基础清洗后 | {stage_stats['after_basic_clean']} |\n")
    f.write(f"| 质量过滤输入 | {quality_input} |\n")
    f.write(f"| 质量过滤后 | {stage_stats['after_quality_clean']} |\n")
    f.write(f"| 质量阶段删除量 | {quality_removed} |\n\n")

    f.write("## 质量规则命中明细\n\n")
    f.write("| 规则Key | 规则 | 命中数 | 在质量阶段命中比例 | 对原始总量删除贡献 |\n")
    f.write("|---|---|---:|---:|---:|\n")
    for item in rule_hits["quality_rules"]:
        f.write(
            f"| {item['rule_key']} | {item['rule']} | {item['hits']} | "
            f"{item['hit_ratio_in_quality_stage_pct']}% | {item['contribution_in_raw_total_pct']}% |\n"
        )

print("\n" + "=" * 70)
print("📊 P2 规则命中统计已输出")
print(f"- {json_out}")
print(f"- {md_out}")
print("=" * 70)