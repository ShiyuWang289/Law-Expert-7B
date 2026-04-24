#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
P3 标准对照数据构建：
- A: 随机500（来自同一 pool）
- B: 工程化500（分层 + 质量评分）

输入：
  raw_data/law_qa_pool.jsonl
输出：
  data/p3/A_random_500.jsonl
  data/p3/B_engineered_500.jsonl
  data/p3/A_random_500_alpaca.json
  data/p3/B_engineered_500_alpaca.json
  data/p3/p3_sampling_report.md
  data/p3/p3_sampling_meta.json
"""

import os
import re
import json
import random
from collections import defaultdict, Counter

INPUT_POOL = "raw_data/law_qa_pool.jsonl"
OUT_DIR = "data/p3"
N = 500
SEED_A = 42
SEED_B = 42  # B组内部采样时也固定seed确保可复现

SYSTEM_PROMPT = "你是一个专业的中国法律顾问，请根据用户的法律问题，给出准确、专业、有法律依据的回答。"
INSTRUCTION = "你是一个法律专家，请根据用户的问题给出专业的回答"


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def to_alpaca(items):
    out = []
    for x in items:
        out.append({
            "instruction": INSTRUCTION,
            "input": x["input"],
            "output": x["output"],
            "system": SYSTEM_PROMPT
        })
    return out


def detect_category(text):
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
    for c, kws in rules:
        if any(k in text for k in kws):
            return c
    return "其他"


def quality_score(item):
    """
    工程化评分：可解释、轻量、稳定
    """
    inp = item.get("input", "").strip()
    out = item.get("output", "").strip()
    s = 0.0

    # 1) 长度质量（适中优先）
    if 10 <= len(inp) <= 180:
        s += 1.0
    if 80 <= len(out) <= 800:
        s += 1.0

    # 2) 法律依据表达
    if re.search(r"第[一二三四五六七八九十百千万0-9]+条", out):
        s += 0.8
    if re.search(r"《[^》]{1,30}法》", out):
        s += 0.8

    # 3) 结构化表达
    if any(k in out for k in ["建议", "依据", "可以", "应当", "责任", "综上"]):
        s += 0.6

    # 4) 噪声惩罚
    if re.search(r"https?://|www\.", out, flags=re.IGNORECASE):
        s -= 1.0

    return s


def stratified_engineered_sample(pool, n, seed=42):
    random.seed(seed)

    buckets = defaultdict(list)
    for x in pool:
        text = x.get("input", "") + " " + x.get("output", "")
        c = detect_category(text)
        buckets[c].append(x)

    total = len(pool)

    # 按类别占比分配配额（至少1）
    alloc = {}
    for c, arr in buckets.items():
        alloc[c] = max(1, int(round(len(arr) / total * n)))

    # 配额和校正到 n
    diff = n - sum(alloc.values())
    cats = sorted(alloc.keys(), key=lambda c: len(buckets[c]), reverse=True)

    i = 0
    while diff != 0 and cats:
        c = cats[i % len(cats)]
        if diff > 0:
            alloc[c] += 1
            diff -= 1
        else:
            if alloc[c] > 1:
                alloc[c] -= 1
                diff += 1
        i += 1

    selected = []
    for c, arr in buckets.items():
        arr_sorted = sorted(arr, key=quality_score, reverse=True)
        take = min(alloc[c], len(arr_sorted))
        selected.extend(arr_sorted[:take])

    # 不足补齐：全局高分补齐
    if len(selected) < n:
        selected_ids = set(id(x) for x in selected)
        rest = [x for x in pool if id(x) not in selected_ids]
        rest_sorted = sorted(rest, key=quality_score, reverse=True)
        selected.extend(rest_sorted[:(n - len(selected))])

    # 超出截断
    selected = selected[:n]
    return selected, alloc, buckets


def category_dist(items):
    cnt = Counter()
    for x in items:
        text = x.get("input", "") + " " + x.get("output", "")
        cnt[detect_category(text)] += 1
    return dict(cnt)


def main():
    if not os.path.exists(INPUT_POOL):
        raise FileNotFoundError(f"未找到候选池文件: {INPUT_POOL}")

    pool = load_jsonl(INPUT_POOL)
    if len(pool) < N:
        raise ValueError(f"候选池样本不足: {len(pool)} < {N}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # A: 随机500
    random.seed(SEED_A)
    A = random.sample(pool, N)

    # B: 工程化500
    B, alloc, buckets = stratified_engineered_sample(pool, N, seed=SEED_B)

    # 保存jsonl
    save_jsonl(f"{OUT_DIR}/A_random_500.jsonl", A)
    save_jsonl(f"{OUT_DIR}/B_engineered_500.jsonl", B)

    # 保存alpaca
    save_json(f"{OUT_DIR}/A_random_500_alpaca.json", to_alpaca(A))
    save_json(f"{OUT_DIR}/B_engineered_500_alpaca.json", to_alpaca(B))

    # 报告
    dist_A = category_dist(A)
    dist_B = category_dist(B)

    meta = {
        "input_pool": INPUT_POOL,
        "pool_size": len(pool),
        "sample_size_each": N,
        "seed_A_random": SEED_A,
        "seed_B_engineered": SEED_B,
        "A_strategy": "random sample from same pool",
        "B_strategy": "stratified sampling by category + quality scoring",
        "engineered_scoring_signals": [
            "input/output length bands",
            "law article citation regex",
            "legal law-name citation regex",
            "structured answer keywords",
            "url noise penalty"
        ],
        "category_alloc_for_B": alloc,
    }
    save_json(f"{OUT_DIR}/p3_sampling_meta.json", meta)

    # markdown报告
    md = []
    md.append("# P3 对照采样报告（A:随机500 vs B:工程化500）\n")
    md.append(f"- 候选池: `{INPUT_POOL}`")
    md.append(f"- 候选池规模: **{len(pool)}**")
    md.append(f"- 每组样本数: **{N}**")
    md.append(f"- A组策略: 随机抽样（seed={SEED_A}）")
    md.append(f"- B组策略: 分层 + 质量评分（seed={SEED_B}）\n")

    md.append("## B组分层配额")
    md.append("| 类别 | 配额 |")
    md.append("|---|---:|")
    for c, v in sorted(alloc.items(), key=lambda x: x[1], reverse=True):
        md.append(f"| {c} | {v} |")

    md.append("\n## A/B类别分布（抽样后）")
    md.append("| 类别 | A随机500 | B工程化500 |")
    md.append("|---|---:|---:|")
    all_cats = sorted(set(dist_A.keys()) | set(dist_B.keys()))
    for c in all_cats:
        md.append(f"| {c} | {dist_A.get(c,0)} | {dist_B.get(c,0)} |")

    md.append("\n## 说明")
    md.append("- A 与 B 均来自同一候选池，保证对照公平。")
    md.append("- A 代表“无工程设计的随机基线”。")
    md.append("- B 代表“有工程设计的样本构建策略”。")
    md.append("- 后续训练必须保持相同配置，仅替换数据集。")

    report_path = f"{OUT_DIR}/p3_sampling_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("=" * 72)
    print("✅ P3 A/B 数据集构建完成")
    print(f"- {OUT_DIR}/A_random_500_alpaca.json")
    print(f"- {OUT_DIR}/B_engineered_500_alpaca.json")
    print(f"- {OUT_DIR}/p3_sampling_report.md")
    print(f"- {OUT_DIR}/p3_sampling_meta.json")
    print("=" * 72)


if __name__ == "__main__":
    main()