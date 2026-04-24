#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用于sft强化
SFT 数据统计看板（可直接运行）
支持输入格式：
1) jsonl: 每行一个 {"input": "...", "output": "..."}（或包含 instruction/system）
2) json : 列表，每项同上

默认优先读取：
- data/law_qa_alpaca.json
- raw_data/law_qa_cleaned.jsonl
你也可以通过 --input 指定文件。
"""

import os
import re
import json
import argparse
from collections import Counter
from statistics import mean

# -----------------------------
# 工具函数
# -----------------------------
def percentile(values, p):
    """简易分位数（0~100）"""
    if not values:
        return 0
    vals = sorted(values)
    k = (len(vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return vals[f]
    return vals[f] + (vals[c] - vals[f]) * (k - f)

def safe_text(x):
    return (x or "").strip()

def normalize_for_dup(text):
    """近重复归一化：去空白、去常见标点、小写"""
    t = text.lower().strip()
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"[，。！？；：、“”‘’（）()《》【】\[\],.!?;:\"\']", "", t)
    return t

def detect_legal_category(text):
    """
    关键词启发式分类（可按你项目再扩展）
    返回一个主类；命中多类时按优先级取第一个
    """
    rules = [
        ("劳动", ["劳动", "工资", "加班", "工伤", "仲裁", "社保", "劳动合同", "辞退"]),
        ("婚姻家庭", ["婚姻", "离婚", "抚养", "彩礼", "夫妻", "家暴", "共同财产", "子女"]),
        ("合同", ["合同", "违约", "解除", "履行", "定金", "赔偿", "债务"]),
        ("交通事故", ["交通事故", "肇事", "酒驾", "交强险", "责任认定"]),
        ("刑事", ["刑法", "犯罪", "量刑", "拘留", "取保候审", "诈骗", "盗窃"]),
        ("房产", ["房产", "房屋", "购房", "过户", "产权", "租赁"]),
        ("公司商事", ["公司", "股东", "法定代表人", "工商", "破产", "清算"]),
        ("民事程序", ["起诉", "诉讼", "管辖", "证据", "判决", "执行", "上诉"]),
        ("行政", ["行政", "行政诉讼", "复议", "行政处罚"]),
    ]
    for cat, kws in rules:
        if any(kw in text for kw in kws):
            return cat
    return "其他"

def has_law_reference(text):
    """
    法条引用检测（启发式）
    如：第XX条、依据《XX法》、根据《XX法》
    """
    patterns = [
        r"第[一二三四五六七八九十百千万0-9]+条",
        r"《[^》]{1,30}法》",
        r"(依据|根据)《[^》]{1,30}》",
    ]
    return any(re.search(p, text) for p in patterns)

def has_url_noise(text):
    return bool(re.search(r"https?://|www\.", text, flags=re.IGNORECASE))

def load_records(input_path):
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".jsonl":
        records = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    if ext == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError("JSON 文件需为列表格式。")

    raise ValueError(f"不支持文件类型: {ext}")

def extract_io(record):
    """
    兼容常见字段：
    - Alpaca: instruction/input/output/system
    - 普通QA: input/output
    """
    inp = safe_text(record.get("input", ""))
    out = safe_text(record.get("output", ""))

    # 如果某些数据把问题放在 instruction
    if not inp and record.get("instruction"):
        inp = safe_text(record.get("instruction", ""))

    return inp, out

# -----------------------------
# 主逻辑
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="输入数据路径（json/jsonl）。默认自动探测。"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="analysis",
        help="输出目录（默认 analysis）"
    )
    args = parser.parse_args()

    # 自动探测输入
    candidates = [
        "data/law_qa_alpaca.json",
        "raw_data/law_qa_cleaned.jsonl",
    ]
    input_path = args.input
    if not input_path:
        for c in candidates:
            if os.path.exists(c):
                input_path = c
                break
    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError(
            "未找到输入文件。请使用 --input 指定，例如：\n"
            "python stats_dashboard.py --input data/law_qa_alpaca.json"
        )

    os.makedirs(args.outdir, exist_ok=True)

    records = load_records(input_path)

    inputs, outputs = [], []
    short_output_cnt = 0
    url_noise_cnt = 0
    law_ref_cnt = 0

    category_counter = Counter()

    # 近重复：分别统计 input / output 归一化后的重复
    norm_in_seen = set()
    norm_out_seen = set()
    dup_in_cnt = 0
    dup_out_cnt = 0

    valid_cnt = 0
    for r in records:
        inp, out = extract_io(r)
        if not inp or not out:
            continue
        valid_cnt += 1

        in_len = len(inp)
        out_len = len(out)

        inputs.append(in_len)
        outputs.append(out_len)

        if out_len < 30:
            short_output_cnt += 1
        if has_url_noise(inp) or has_url_noise(out):
            url_noise_cnt += 1
        if has_law_reference(out):
            law_ref_cnt += 1

        cat = detect_legal_category(inp + " " + out)
        category_counter[cat] += 1

        n_in = normalize_for_dup(inp)
        n_out = normalize_for_dup(out)
        if n_in in norm_in_seen:
            dup_in_cnt += 1
        else:
            norm_in_seen.add(n_in)

        if n_out in norm_out_seen:
            dup_out_cnt += 1
        else:
            norm_out_seen.add(n_out)

    if valid_cnt == 0:
        raise RuntimeError("有效样本为 0（input/output 为空）。请检查数据格式。")

    def pct(x):
        return round(100.0 * x / valid_cnt, 2)

    stats = {
        "input_file": input_path,
        "valid_samples": valid_cnt,
        "input_length": {
            "mean": round(mean(inputs), 2),
            "p50": round(percentile(inputs, 50), 2),
            "p95": round(percentile(inputs, 95), 2),
            "max": max(inputs),
        },
        "output_length": {
            "mean": round(mean(outputs), 2),
            "p50": round(percentile(outputs, 50), 2),
            "p95": round(percentile(outputs, 95), 2),
            "max": max(outputs),
        },
        "quality_indicators": {
            "output_lt_30_count": short_output_cnt,
            "output_lt_30_ratio_pct": pct(short_output_cnt),
            "url_noise_count": url_noise_cnt,
            "url_noise_ratio_pct": pct(url_noise_cnt),
            "law_reference_count": law_ref_cnt,
            "law_reference_ratio_pct": pct(law_ref_cnt),
            "duplicate_input_count": dup_in_cnt,
            "duplicate_input_ratio_pct": pct(dup_in_cnt),
            "duplicate_output_count": dup_out_cnt,
            "duplicate_output_ratio_pct": pct(dup_out_cnt),
        },
        "category_distribution": dict(category_counter.most_common()),
    }

    # 1) JSON
    json_path = os.path.join(args.outdir, "sft_data_stats.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 2) CSV（类别分布）
    csv_path = os.path.join(args.outdir, "category_distribution.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("category,count,ratio_pct\n")
        for k, v in category_counter.most_common():
            f.write(f"{k},{v},{pct(v)}\n")

    # 3) Markdown 看板
    md_path = os.path.join(args.outdir, "sft_data_dashboard.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# SFT 数据统计看板\n\n")
        f.write(f"- 输入文件: `{input_path}`\n")
        f.write(f"- 有效样本数: **{valid_cnt}**\n\n")

        f.write("## 长度分布\n\n")
        f.write("| 指标 | input | output |\n")
        f.write("|---|---:|---:|\n")
        f.write(f"| mean | {stats['input_length']['mean']} | {stats['output_length']['mean']} |\n")
        f.write(f"| P50 | {stats['input_length']['p50']} | {stats['output_length']['p50']} |\n")
        f.write(f"| P95 | {stats['input_length']['p95']} | {stats['output_length']['p95']} |\n")
        f.write(f"| max | {stats['input_length']['max']} | {stats['output_length']['max']} |\n\n")

        q = stats["quality_indicators"]
        f.write("## 质量与风险指标\n\n")
        f.write("| 指标 | 数值 |\n")
        f.write("|---|---:|\n")
        f.write(f"| output < 30 数量 | {q['output_lt_30_count']} |\n")
        f.write(f"| output < 30 占比(%) | {q['output_lt_30_ratio_pct']} |\n")
        f.write(f"| URL 噪声数量 | {q['url_noise_count']} |\n")
        f.write(f"| URL 噪声占比(%) | {q['url_noise_ratio_pct']} |\n")
        f.write(f"| 法条引用数量 | {q['law_reference_count']} |\n")
        f.write(f"| 法条引用占比(%) | {q['law_reference_ratio_pct']} |\n")
        f.write(f"| 近重复 input 数量 | {q['duplicate_input_count']} |\n")
        f.write(f"| 近重复 input 占比(%) | {q['duplicate_input_ratio_pct']} |\n")
        f.write(f"| 近重复 output 数量 | {q['duplicate_output_count']} |\n")
        f.write(f"| 近重复 output 占比(%) | {q['duplicate_output_ratio_pct']} |\n\n")

        f.write("## 法律类别分布（启发式）\n\n")
        f.write("| 类别 | 数量 | 占比(%) |\n")
        f.write("|---|---:|---:|\n")
        for k, v in category_counter.most_common():
            f.write(f"| {k} | {v} | {pct(v)} |\n")

    # 控制台打印摘要
    print("=" * 70)
    print("✅ SFT 数据统计完成")
    print(f"输入文件: {input_path}")
    print(f"有效样本: {valid_cnt}")
    print("-" * 70)
    print("长度分布:")
    print(f"  input  mean={stats['input_length']['mean']}  P50={stats['input_length']['p50']}  P95={stats['input_length']['p95']}  max={stats['input_length']['max']}")
    print(f"  output mean={stats['output_length']['mean']} P50={stats['output_length']['p50']} P95={stats['output_length']['p95']} max={stats['output_length']['max']}")
    print("-" * 70)
    print("质量指标:")
    print(f"  output<30 占比: {q['output_lt_30_ratio_pct']}%")
    print(f"  URL噪声占比: {q['url_noise_ratio_pct']}%")
    print(f"  法条引用占比: {q['law_reference_ratio_pct']}%")
    print(f"  近重复(input)占比: {q['duplicate_input_ratio_pct']}%")
    print(f"  近重复(output)占比: {q['duplicate_output_ratio_pct']}%")
    print("-" * 70)
    print(f"输出文件:\n- {json_path}\n- {csv_path}\n- {md_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()