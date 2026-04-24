#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import csv
from collections import defaultdict

# ========== 配置 ==========
QUESTIONS_PATH = "eval/p3_eval_questions.json"
ANS_A_PATH = "eval/answers_A.json"   # 你先生成：[{id,answer}]
ANS_B_PATH = "eval/answers_B.json"   # 你先生成：[{id,answer}]
OUT_CSV = "eval/p3_auto_eval_scores.csv"
OUT_MD = "eval/p3_auto_eval_report.md"

LAW_PAT = re.compile(r"第[一二三四五六七八九十百千万0-9]+条|《[^》]{1,30}法》")
STRUCT_KWS = ["建议", "依据", "结论", "可以", "应当", "流程", "步骤", "证据", "起诉", "仲裁", "复议", "诉讼"]
ACTION_KWS = ["建议", "尽快", "先", "准备", "保留证据", "咨询律师", "起诉", "申请", "仲裁", "复议"]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def score_answer(ans: str):
    ans = (ans or "").strip()
    if not ans:
        return {"law_basis":0, "structure":0, "coverage":0, "length":0, "total":0}

    # 1) 法律依据（0-3）
    law_hits = len(LAW_PAT.findall(ans))
    law_basis = 3 if law_hits >= 2 else (2 if law_hits == 1 else 0)

    # 2) 结构化（0-3）
    struct_hits = sum(1 for k in STRUCT_KWS if k in ans)
    bullet_like = bool(re.search(r"[一二三四五六七八九十]\s*[、\.]|^\d+[\.、)]", ans, flags=re.M))
    structure = 1
    if struct_hits >= 3: structure += 1
    if bullet_like: structure += 1
    structure = min(structure, 3)

    # 3) 覆盖度（0-3）：结论+依据+行动建议
    has_conclusion = any(k in ans for k in ["可以", "应当", "通常", "一般", "结论"])
    has_basis = law_hits > 0 or "依据" in ans
    has_action = any(k in ans for k in ACTION_KWS)
    coverage = int(has_conclusion) + int(has_basis) + int(has_action)  # 0-3

    # 4) 长度有效性（0-1）
    length = 1 if 80 <= len(ans) <= 1200 else 0

    total = law_basis + structure + coverage + length  # 0-10
    return {
        "law_basis": law_basis,
        "structure": structure,
        "coverage": coverage,
        "length": length,
        "total": total
    }

def main():
    questions = load_json(QUESTIONS_PATH)
    ans_a = {x["id"]: x["answer"] for x in load_json(ANS_A_PATH)}
    ans_b = {x["id"]: x["answer"] for x in load_json(ANS_B_PATH)}

    rows = []
    sum_a = defaultdict(float)
    sum_b = defaultdict(float)

    for q in questions:
        qid = q["id"]
        qa = ans_a.get(qid, "")
        qb = ans_b.get(qid, "")

        sa = score_answer(qa)
        sb = score_answer(qb)

        winner = "Tie"
        if sa["total"] > sb["total"]:
            winner = "A"
        elif sb["total"] > sa["total"]:
            winner = "B"

        row = {
            "id": qid,
            "category": q.get("category",""),
            "question": q["question"],
            "A_total": sa["total"],
            "B_total": sb["total"],
            "A_law_basis": sa["law_basis"],
            "B_law_basis": sb["law_basis"],
            "A_structure": sa["structure"],
            "B_structure": sb["structure"],
            "A_coverage": sa["coverage"],
            "B_coverage": sb["coverage"],
            "A_length": sa["length"],
            "B_length": sb["length"],
            "winner": winner
        }
        rows.append(row)

        for k,v in sa.items():
            sum_a[k] += v
        for k,v in sb.items():
            sum_b[k] += v

    n = len(rows)
    avg_a = {k: round(v/n, 4) for k,v in sum_a.items()}
    avg_b = {k: round(v/n, 4) for k,v in sum_b.items()}
    win_a = sum(1 for r in rows if r["winner"]=="A")
    win_b = sum(1 for r in rows if r["winner"]=="B")
    tie = sum(1 for r in rows if r["winner"]=="Tie")

    os.makedirs("eval", exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md = []
    md.append("# P3 自动评测报告（A:随机500 vs B:工程化500）\n")
    md.append("## 训练指标")
    md.append("- A eval_loss: **1.1755**")
    md.append("- B eval_loss: **1.0844**")
    md.append("- 相对下降: **7.75%**\n")

    md.append("## 自动评测均分（0-10 total）")
    md.append(f"- A 平均总分: **{avg_a['total']}**")
    md.append(f"- B 平均总分: **{avg_b['total']}**")
    md.append(f"- 胜场: A={win_a}, B={win_b}, Tie={tie}\n")

    md.append("## 维度均分")
    md.append("| 维度 | A | B |")
    md.append("|---|---:|---:|")
    for k in ["law_basis","structure","coverage","length","total"]:
        md.append(f"| {k} | {avg_a[k]} | {avg_b[k]} |")

    md.append("\n## 结论")
    if avg_b["total"] > avg_a["total"]:
        md.append("- B 在自动评测总分上优于 A，与 eval_loss 结论一致，支持“工程化采样有效”。")
    else:
        md.append("- 自动评测未显著支持 B 优势，建议扩大题量或补充人工评测复核。")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"✅ 写出: {OUT_CSV}")
    print(f"✅ 写出: {OUT_MD}")

if __name__ == "__main__":
    main()