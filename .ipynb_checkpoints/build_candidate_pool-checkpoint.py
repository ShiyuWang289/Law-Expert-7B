#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从原始 DISC-Law-SFT 生成“可对照 pool”（未精选）
用途：为 P3 构建 A 随机500 vs B 工程化500 的共同候选池

默认输入：
./raw_data/DISC-Law-SFT/DISC-Law-SFT-Pair-QA-released.jsonl

默认输出：
- raw_data/law_qa_pool.jsonl           # 统一格式: {"input","output"}
- analysis/pool_stats.json             # 统计
- analysis/pool_build_report.md        # 报告
"""

import os
import re
import json
import argparse
from collections import Counter


def pct(x, n):
    return round((x / n * 100), 2) if n else 0.0


def detect_fields(obj):
    """
    兼容不同字段命名
    优先 input/output；否则尝试 question/answer / prompt/response
    """
    if "input" in obj and "output" in obj:
        return (obj.get("input", ""), obj.get("output", ""))
    if "question" in obj and "answer" in obj:
        return (obj.get("question", ""), obj.get("answer", ""))
    if "prompt" in obj and "response" in obj:
        return (obj.get("prompt", ""), obj.get("response", ""))
    return ("", "")


def normalize_text(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def build_pool(input_path, output_path, min_in_len=5, min_out_len=20, max_total_len=2000):
    """
    只做“基础可用性清洗”，避免过强过滤导致候选池过小
    """
    counters = Counter()

    seen_inputs = set()
    kept = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            counters["raw_total"] += 1

            try:
                obj = json.loads(line)
            except Exception:
                counters["bad_json"] += 1
                continue

            inp, out = detect_fields(obj)
            inp = normalize_text(inp)
            out = normalize_text(out)

            # R0: 空值
            if not inp or not out:
                counters["drop_empty"] += 1
                continue

            # R1: 基础长度（宽松）
            if len(inp) < min_in_len:
                counters["drop_input_too_short"] += 1
                continue
            if len(out) < min_out_len:
                counters["drop_output_too_short"] += 1
                continue

            # R2: 总长度上限（宽松，主要防极端长样本）
            if len(inp) + len(out) > max_total_len:
                counters["drop_total_too_long"] += 1
                continue

            # R3: 精确去重（仅按 input 去重）
            if inp in seen_inputs:
                counters["drop_dup_input"] += 1
                continue
            seen_inputs.add(inp)

            # 可选：仅过滤明显噪声链接（宽松）
            if re.search(r"https?://|www\.", out, flags=re.IGNORECASE):
                counters["drop_contains_url"] += 1
                continue

            kept.append({"input": inp, "output": out})

    counters["kept"] = len(kept)

    # 输出 pool
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return counters


def write_reports(counters, input_path, output_path, report_dir="analysis"):
    os.makedirs(report_dir, exist_ok=True)

    stats = {
        "input_file": input_path,
        "output_pool_file": output_path,
        "raw_total": counters["raw_total"],
        "kept": counters["kept"],
        "drop": {
            "bad_json": counters["bad_json"],
            "empty": counters["drop_empty"],
            "input_too_short": counters["drop_input_too_short"],
            "output_too_short": counters["drop_output_too_short"],
            "total_too_long": counters["drop_total_too_long"],
            "dup_input": counters["drop_dup_input"],
            "contains_url": counters["drop_contains_url"],
        }
    }

    json_path = os.path.join(report_dir, "pool_stats.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(report_dir, "pool_build_report.md")
    raw_total = counters["raw_total"]
    kept = counters["kept"]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 候选池构建报告（P3前置）\n\n")
        f.write(f"- 输入文件: `{input_path}`\n")
        f.write(f"- 输出文件: `{output_path}`\n\n")
        f.write("## 总体结果\n\n")
        f.write(f"- 原始样本数: **{raw_total}**\n")
        f.write(f"- 候选池样本数: **{kept}**\n")
        f.write(f"- 保留率: **{pct(kept, raw_total)}%**\n\n")

        f.write("## 过滤明细\n\n")
        f.write("| 过滤项 | 数量 | 占原始比例 |\n")
        f.write("|---|---:|---:|\n")
        for k, label in [
            ("bad_json", "JSON解析失败"),
            ("drop_empty", "空值过滤"),
            ("drop_input_too_short", "输入过短"),
            ("drop_output_too_short", "输出过短"),
            ("drop_total_too_long", "总长度过长"),
            ("drop_dup_input", "input去重"),
            ("drop_contains_url", "URL噪声"),
        ]:
            v = counters[k]
            f.write(f"| {label} | {v} | {pct(v, raw_total)}% |\n")

        f.write("\n## 说明\n\n")
        f.write("- 本阶段仅做“基础可用性清洗”，不做工程化精选。\n")
        f.write("- 该 pool 用于后续 A/B 对照：\n")
        f.write("  - A：随机500\n")
        f.write("  - B：清洗+分层精选500\n")
        f.write("- 要求 A/B 来自同一个 pool，保证对照公平。\n")

    return json_path, md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="./raw_data/DISC-Law-SFT/DISC-Law-SFT-Pair-QA-released.jsonl",
        help="原始 DISC-Law-SFT jsonl 文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="raw_data/law_qa_pool.jsonl",
        help="输出候选池 jsonl 文件路径"
    )
    parser.add_argument("--min_in_len", type=int, default=5)
    parser.add_argument("--min_out_len", type=int, default=20)
    parser.add_argument("--max_total_len", type=int, default=2000)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"未找到输入文件: {args.input}")

    counters = build_pool(
        input_path=args.input,
        output_path=args.output,
        min_in_len=args.min_in_len,
        min_out_len=args.min_out_len,
        max_total_len=args.max_total_len
    )

    json_path, md_path = write_reports(
        counters=counters,
        input_path=args.input,
        output_path=args.output,
        report_dir="analysis"
    )

    print("=" * 72)
    print("✅ 候选池构建完成")
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"原始样本: {counters['raw_total']}")
    print(f"候选池样本: {counters['kept']}")
    print(f"报告: {json_path}")
    print(f"报告: {md_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()