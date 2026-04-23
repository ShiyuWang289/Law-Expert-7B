"""
Step 5: RAG效果评测
对比 无RAG vs 有RAG 在你的10个测试案例上的表现。
复用 prompt_engineering 中已有的评测逻辑。
"""

import sys
import json
import time
from openai import OpenAI

# 复用已有的评测器
sys.path.append("../prompt_engineering")
from evaluator import evaluate_single_response
from test_cases import TEST_CASES
from retriever import LawRetriever
from rag_pipeline import (
    ask_with_rag, ask_without_rag,
    BASE_SYSTEM_PROMPT, MODEL_NAME, VLLM_BASE_URL
)

RESULTS_DIR = "./results"

def run_comparison():
    """运行 无RAG vs 有RAG 的完整对比实验"""
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)

    retriever = LawRetriever()
    client    = OpenAI(api_key="EMPTY", base_url=VLLM_BASE_URL)

    all_results = []
    total = len(TEST_CASES) * 2  # 两种模式
    current = 0

    for mode in ["no_rag", "with_rag"]:
        print(f"\n{'='*60}")
        print(f"测试模式: {mode}")
        print(f"{'='*60}")

        for tc in TEST_CASES:
            current += 1
            print(f"[{current}/{total}] {tc['id']} - {tc['category']}")

            if mode == "no_rag":
                raw = ask_without_rag(tc["question"], client)
            else:
                raw = ask_with_rag(
                    tc["question"], retriever, client,
                    top_k=3, verbose=False
                )

            # 复用已有评测逻辑
            eval_result = evaluate_single_response(
                raw["answer"], tc, mode
            )
            eval_result["latency"]       = raw["latency"]
            eval_result["rag_used"]      = raw.get("rag_used", False)
            eval_result["retrieved_laws"] = raw.get("retrieved_laws", [])

            all_results.append(eval_result)

            score    = eval_result["scores"]["total"]
            kp_rate  = eval_result["details"]["key_points"]["coverage_rate"]
            print(f"  总分: {score:.1f}/100 | 关键点覆盖: {kp_rate:.0%} | "
                  f"耗时: {raw['latency']:.1f}s")

            time.sleep(0.3)

    return all_results


def generate_rag_report(results: list) -> str:
    """生成RAG对比报告"""
    def avg(lst): return sum(lst)/len(lst) if lst else 0

    no_rag   = [r for r in results if r["prompt_version"] == "no_rag"]
    with_rag = [r for r in results if r["prompt_version"] == "with_rag"]

    report = ["# RAG vs 无RAG 对比评测报告\n"]

    # 总体对比
    report.append("## 总体对比\n")
    report.append("| 指标 | 无RAG | 有RAG | 变化 |")
    report.append("|:---|:---:|:---:|:---:|")

    metrics = [
        ("scores.total",                        "平均总分"),
        ("details.key_points.coverage_rate",    "关键点覆盖率"),
        ("scores.citation_format",              "法条格式得分"),
        ("latency",                             "平均响应时间(s)"),
    ]

    for key, name in metrics:
        def get_val(r, k):
            keys = k.split(".")
            v = r
            for kk in keys:
                v = v[kk]
            return v

        no_vals   = [get_val(r, key) for r in no_rag]
        with_vals = [get_val(r, key) for r in with_rag]
        no_avg    = avg(no_vals)
        with_avg  = avg(with_vals)
        delta     = with_avg - no_avg
        sign      = "+" if delta >= 0 else ""

        if "rate" in key:
            report.append(f"| {name} | {no_avg:.1%} | {with_avg:.1%} | {sign}{delta:.1%} |")
        elif "time" in name or "latency" in key:
            report.append(f"| {name} | {no_avg:.1f}s | {with_avg:.1f}s | {sign}{delta:.1f}s |")
        else:
            report.append(f"| {name} | {no_avg:.1f} | {with_avg:.1f} | {sign}{delta:.1f} |")

    # 按类别对比
    report.append("\n## 按问题类别对比\n")
    report.append("| 问题类别 | 无RAG | 有RAG | 检索到的法条 |")
    report.append("|:---|:---:|:---:|:---|")

    categories = sorted(set(tc["category"] for tc in TEST_CASES))
    for cat in categories:
        no_scores   = [r["scores"]["total"] for r in no_rag   if r["category"] == cat]
        with_scores = [r["scores"]["total"] for r in with_rag if r["category"] == cat]
        retrieved   = [r.get("retrieved_laws", []) for r in with_rag if r["category"] == cat]
        retrieved_str = ", ".join(retrieved[0]) if retrieved and retrieved[0] else "无"

        no_avg   = avg(no_scores)
        with_avg = avg(with_scores)
        delta    = with_avg - no_avg
        sign     = "↑" if delta > 0 else ("↓" if delta < 0 else "→")

        report.append(f"| {cat} | {no_avg:.1f} | {with_avg:.1f} | {sign}{abs(delta):.1f} | {retrieved_str} |")

    # RAG的局限性分析
    report.append("\n## RAG局限性分析\n")
    degraded = [
        r for r in with_rag
        if r["scores"]["total"] < next(
            (nr["scores"]["total"] for nr in no_rag if nr["test_id"] == r["test_id"]),
            100
        )
    ]
    if degraded:
        report.append("以下案例在加入RAG后分数下降，需要关注：\n")
        for r in degraded:
            no_score = next(
                nr["scores"]["total"] for nr in no_rag if nr["test_id"] == r["test_id"]
            )
            report.append(f"- **{r['test_id']}** ({r['category']}): "
                         f"{no_score:.1f} → {r['scores']['total']:.1f} "
                         f"({r['scores']['total']-no_score:.1f})")

    return "\n".join(report)


if __name__ == "__main__":
    print("🔬 开始 RAG 对比评测实验")
    results = run_comparison()

    # 保存原始结果
    with open(f"{RESULTS_DIR}/rag_raw_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 生成报告
    report = generate_rag_report(results)
    report_path = f"{RESULTS_DIR}/rag_eval_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n📊 报告已保存到: {report_path}")
    print("\n" + "="*60)
    print(report[:3000])
