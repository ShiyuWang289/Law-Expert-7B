#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法：
python eval/regression_check.py --run_tag run_20260426_220000
"""
import os, json, argparse

THRESHOLDS = {
    "coverage_drop": 0.02,         # 覆盖率下降超过 2pct 告警
    "law_acc_drop": 0.02,          # 法条准确率下降超过 2pct 告警
    "repetition_increase": 0.05,   # 重复率上升超过 5pct 告警
    "hallucination_increase": 2    # 幻觉条数增加超过 2 告警
}

def load(p):
    return json.load(open(p, "r", encoding="utf-8")) if os.path.exists(p) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_tag", required=True)
    args = ap.parse_args()

    d = f"eval/runs/{args.run_tag}"
    base = load(f"{d}/base_summary.json")
    sft  = load(f"{d}/sft_summary.json")
    dpo  = load(f"{d}/dpo_summary.json")
    if not (base and sft and dpo):
        raise SystemExit("缺少 base/sft/dpo summary 文件")

    alerts = []
    def check(new, old, name):
        if old["coverage"] - new["coverage"] > THRESHOLDS["coverage_drop"]:
            alerts.append(f"[ALERT] {name}: coverage drop > 2pct")
        if old["law_accuracy"] - new["law_accuracy"] > THRESHOLDS["law_acc_drop"]:
            alerts.append(f"[ALERT] {name}: law_accuracy drop > 2pct")
        if new["repetition_rate"] - old["repetition_rate"] > THRESHOLDS["repetition_increase"]:
            alerts.append(f"[ALERT] {name}: repetition_rate increase > 5pct")
        if new["hallucination_count"] - old["hallucination_count"] > THRESHOLDS["hallucination_increase"]:
            alerts.append(f"[ALERT] {name}: hallucination_count increase > 2")

    check(sft, base, "SFT vs BASE")
    check(dpo, sft, "DPO vs SFT")

    report_md = f"eval/reports/regression_{args.run_tag}.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(f"# Regression Report ({args.run_tag})\n\n")
        f.write("| Model | law_accuracy | coverage | repetition_rate | avg_len | hallucination_count |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for x in [base, sft, dpo]:
            f.write(f"| {x['model']} | {x['law_accuracy']:.3f} | {x['coverage']:.3f} | {x['repetition_rate']:.3f} | {x['avg_len']:.1f} | {x['hallucination_count']} |\n")
        f.write("\n## Alerts\n")
        if alerts:
            for a in alerts:
                f.write(f"- {a}\n")
        else:
            f.write("- ✅ No regression alert.\n")

    print("saved:", report_md)
    if alerts:
        print("\n".join(alerts))
    else:
        print("✅ No regression alert.")

if __name__ == "__main__":
    main()