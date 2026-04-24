#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
读取 /root/autodl-tmp/saves 下的 P4 三组结果并输出汇总。
支持两种来源：
1) metrics 文件（trainer_state.json / all_results.json / train_results.json）
2) 手工给定结果（fallback，避免跑通问题）
"""

import os
import json
from typing import Dict, Any

# 你可以按实际目录改这三个路径
RUNS = {
    "5e-5": "/root/autodl-tmp/saves/p4_lr_5e5",
    "1e-4": "/root/autodl-tmp/saves/p4_lr_1e4",
    "2e-4": "/root/autodl-tmp/saves/p4_lr_2e4",
}

# 你刚给出的结果，作为兜底（即使文件解析失败也能产出）
FALLBACK = {
    "2e-4": {
        "epoch": 0.3333333333333333,
        "eval_loss": 1.068247675895691,
        "eval_runtime": 9.3164,
        "eval_samples_per_second": 1.288,
        "eval_steps_per_second": 0.644,
        "total_flos": 576843452989440,
        "train_loss": 1.108687400817871,
        "train_runtime": 77.8786,
        "train_samples_per_second": 0.416,
        "train_steps_per_second": 0.116
    },
    "1e-4": {
        "epoch": 0.3333333333333333,
        "eval_loss": 1.0823501348495483,
        "eval_runtime": 9.3213,
        "eval_samples_per_second": 1.287,
        "eval_steps_per_second": 0.644,
        "total_flos": 576843452989440,
        "train_loss": 1.1218620936075847,
        "train_runtime": 77.8653,
        "train_samples_per_second": 0.416,
        "train_steps_per_second": 0.116
    },
    "5e-5": {
        "epoch": 0.3333333333333333,
        "eval_loss": 1.1024996042251587,
        "eval_runtime": 9.3142,
        "eval_samples_per_second": 1.288,
        "eval_steps_per_second": 0.644,
        "total_flos": 576843452989440,
        "train_loss": 1.1398625903659396,
        "train_runtime": 77.8442,
        "train_samples_per_second": 0.416,
        "train_steps_per_second": 0.116
    }
}

def read_json(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def extract_from_trainer_state(d: Dict[str, Any]):
    out = {}
    # best eval
    out["eval_loss"] = d.get("best_metric", None)

    # 从 log_history 抓最后一次 train/eval
    log_history = d.get("log_history", [])
    last_train = None
    last_eval = None
    for x in log_history:
        if "loss" in x:
            last_train = x
        if "eval_loss" in x:
            last_eval = x

    if last_train:
        out["train_loss"] = last_train.get("loss")
        out["train_runtime"] = last_train.get("train_runtime", None)  # 通常没有
    if last_eval:
        out["eval_loss"] = last_eval.get("eval_loss", out.get("eval_loss"))
        out["eval_runtime"] = last_eval.get("eval_runtime")
        out["eval_samples_per_second"] = last_eval.get("eval_samples_per_second")
        out["eval_steps_per_second"] = last_eval.get("eval_steps_per_second")
        out["epoch"] = last_eval.get("epoch")

    return out

def load_run_metrics(run_dir: str):
    # 优先 all_results.json
    all_results = read_json(os.path.join(run_dir, "all_results.json"))
    if all_results:
        return {
            "train_loss": all_results.get("train_loss"),
            "eval_loss": all_results.get("eval_loss"),
            "train_runtime": all_results.get("train_runtime"),
            "eval_runtime": all_results.get("eval_runtime"),
            "eval_samples_per_second": all_results.get("eval_samples_per_second"),
            "eval_steps_per_second": all_results.get("eval_steps_per_second"),
            "epoch": all_results.get("epoch")
        }

    # 次选 trainer_state.json
    ts = read_json(os.path.join(run_dir, "trainer_state.json"))
    if ts:
        return extract_from_trainer_state(ts)

    # 再次选 train_results.json
    tr = read_json(os.path.join(run_dir, "train_results.json"))
    if tr:
        return {
            "train_loss": tr.get("train_loss"),
            "train_runtime": tr.get("train_runtime"),
            "epoch": tr.get("epoch")
        }

    return None

def fmt(x, nd=4):
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)

def main():
    rows = []
    for lr, run_dir in RUNS.items():
        m = load_run_metrics(run_dir)
        if not m:
            m = FALLBACK.get(lr, {})
        m["lr"] = lr
        rows.append(m)

    # 按 eval_loss 排序（越小越好）
    rows_sorted = sorted(rows, key=lambda r: r.get("eval_loss", 1e9))

    print("=== P4 LR Ablation Summary ===")
    print("lr, train_loss, eval_loss, train_runtime, eval_runtime")
    for r in rows_sorted:
        print(f"{r['lr']}, {fmt(r.get('train_loss'))}, {fmt(r.get('eval_loss'))}, {fmt(r.get('train_runtime'))}, {fmt(r.get('eval_runtime'))}")

    best = rows_sorted[0]
    print(f"\nRECOMMENDED_LR = {best['lr']} (lowest eval_loss={fmt(best.get('eval_loss'))})")

    # 写 markdown 报告
    os.makedirs("analysis", exist_ok=True)
    md_path = "analysis/p4_lr_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# P4 Learning Rate Ablation Report\n\n")
        f.write("| learning rate | train_loss | eval_loss | train_runtime(s) | eval_runtime(s) |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for r in rows_sorted:
            f.write(f"| {r['lr']} | {fmt(r.get('train_loss'))} | {fmt(r.get('eval_loss'))} | {fmt(r.get('train_runtime'))} | {fmt(r.get('eval_runtime'))} |\n")
        f.write(f"\n**Recommended LR:** `{best['lr']}` (lowest eval_loss={fmt(best.get('eval_loss'))})\n")

    print(f"✅ Markdown report saved to: {md_path}")

if __name__ == "__main__":
    main()