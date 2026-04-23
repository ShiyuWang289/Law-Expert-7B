# phase_d/ablation_with_stats.py
"""
改进的消融实验：添加多次运行和统计分析
"""

import sys
import os
import json
import time
from pathlib import Path
from openai import OpenAI
import numpy as np

PHASE_D_DIR = Path(__file__).parent.absolute()
RAG_DIR = PHASE_D_DIR.parent
LLAMA_FACTORY_ROOT = RAG_DIR.parent

sys.path.insert(0, str(LLAMA_FACTORY_ROOT))
sys.path.insert(0, str(RAG_DIR))

from prompt_engineering.evaluator import evaluate_single_response
from prompt_engineering.test_cases import TEST_CASES
from phase_c.enterprise_pipeline import EnterpriseLawRAG, BASE_SYSTEM_PROMPT
from retriever import LawRetriever
from rag_pipeline import ask_without_rag, ask_with_rag

RESULTS_DIR = PHASE_D_DIR / "results"
VLLM_BASE_URL = "http://localhost:6006/v1"
MODEL_NAME = "law-expert"


def run_single_ablation(simple_retriever, enterprise_rag, client, chunks_v3):
    """运行一次完整的消融实验"""
    all_results = []
    versions = ["V1_no_rag", "V2_simple_rag", "V3_better_chunking",
                "V4_hybrid_reranker", "V5_full_citation"]

    for version in versions:
        if version == "V2_simple_rag" and not simple_retriever:
            continue

        for tc in TEST_CASES:
            try:
                if version == "V1_no_rag":
                    raw = ask_without_rag(tc["question"], client, verbose=False)
                elif version == "V2_simple_rag":
                    raw = ask_with_rag(tc["question"], simple_retriever, client,
                                      top_k=3, verbose=False)
                elif version in ["V3_better_chunking", "V4_hybrid_reranker",
                                "V5_full_citation"]:
                    output = enterprise_rag.answer(tc["question"], top_k=3, verbose=False)
                    raw = {
                        "question": tc["question"],
                        "answer":   output["answer"],
                        "latency":  output["latency_s"],
                        "rag_used": True,
                        "citations": output.get("citations", []),
                        "citation_stats": output.get("citation_stats", {}),
                    }

                eval_result = evaluate_single_response(raw["answer"], tc, version)
                eval_result["latency"] = raw["latency"]
                all_results.append(eval_result)

            except Exception as e:
                print(f"  ⚠️  {tc['id']} - {version} 处理失败: {str(e)[:50]}")
                continue

    return all_results


def run_multiple_times(num_runs: int = 3):
    """多次运行并统计"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key="EMPTY", base_url=VLLM_BASE_URL)

    # 加载chunks
    chunks_path = RAG_DIR / "phase_a" / "chunk_results" / "chunks_C.json"
    with open(chunks_path, encoding="utf-8") as f:
        chunks_v3 = json.load(f)

    # 初始化检索器
    vector_store_path = RAG_DIR / "vector_store"
    try:
        simple_retriever = LawRetriever(
            vector_dir=str(vector_store_path),
            model_cache="/root/autodl-tmp/embedding_model",
            model_name="BAAI/bge-small-zh-v1.5"
        )
    except Exception as e:
        print(f"⚠️  V2初始化失败: {e}")
        simple_retriever = None

    try:
        enterprise_rag = EnterpriseLawRAG(chunks_v3, offline_mode=True)
    except Exception as e:
        print(f"❌ 企业级RAG初始化失败: {e}")
        raise

    # 多次运行
    all_runs = []
    for run_idx in range(num_runs):
        print(f"\n{'='*60}")
        print(f"运行 {run_idx + 1}/{num_runs}")
        print(f"{'='*60}")
        
        results = run_single_ablation(simple_retriever, enterprise_rag, client, chunks_v3)
        all_runs.append(results)

    # 统计分析
    print(f"\n{'='*60}")
    print(f"统计分析（{num_runs}次运行）")
    print(f"{'='*60}\n")

    versions = ["V1_no_rag", "V2_simple_rag", "V3_better_chunking",
                "V4_hybrid_reranker", "V5_full_citation"]

    report = ["# 消融实验统计分析\n"]
    report.append(f"## 实验配置\n")
    report.append(f"- 运行次数: {num_runs}\n")
    report.append(f"- 每次测试用例: {len(TEST_CASES)}\n")
    report.append(f"- 总计评估次数: {num_runs * len(TEST_CASES) * len(versions)}\n\n")

    report.append("## 版本性能对比（含置信区间）\n")
    report.append("| 版本 | 平均分 | 标准差 | 95% CI | 最好 | 最差 |\n")
    report.append("|:---|:---:|:---:|:---:|:---:|:---:|\n")

    stats_by_version = {}

    for v in versions:
        # 收集所有运行中该版本的分数
        scores_all_runs = []
        for run_results in all_runs:
            v_results = [r["scores"]["total"] for r in run_results if r.get("prompt_version") == v]
            if v_results:
                scores_all_runs.append(np.mean(v_results))

        if not scores_all_runs:
            report.append(f"| {v} | N/A | N/A | N/A | N/A | N/A |\n")
            continue

        scores_all_runs = np.array(scores_all_runs)
        mean = np.mean(scores_all_runs)
        std = np.std(scores_all_runs)
        sem = std / np.sqrt(len(scores_all_runs))  # 标准误
        ci95 = 1.96 * sem  # 95% 置信区间

        stats_by_version[v] = {
            "mean": float(mean),
            "std": float(std),
            "ci95": float(ci95),
            "min": float(np.min(scores_all_runs)),
            "max": float(np.max(scores_all_runs)),
        }

        report.append(
            f"| {v} | {mean:.1f} | {std:.2f} | ±{ci95:.1f} | {np.max(scores_all_runs):.1f} | {np.min(scores_all_runs):.1f} |\n"
        )

    # 保存原始数据和统计
    with open(RESULTS_DIR / "ablation_multiple_runs.json", "w", encoding="utf-8") as f:
        json.dump({
            "num_runs": num_runs,
            "all_results": all_runs,
            "stats_by_version": stats_by_version,
        }, f, ensure_ascii=False, indent=2)

    report.append("\n## 性能稳定性分析\n")
    report.append("| 版本 | 稳定性评价 | 建议 |\n")
    report.append("|:---|:---|:---:|\n")

    for v in versions:
        if v not in stats_by_version:
            continue
        
        cv = stats_by_version[v]["std"] / stats_by_version[v]["mean"]  # 变异系数
        
        if cv < 0.05:
            stability = "✅ 非常稳定"
        elif cv < 0.10:
            stability = "⚠️ 较稳定"
        else:
            stability = "❌ 不稳定"
        
        report.append(f"| {v} | {stability} (CV={cv:.3f}) | {'信任结果' if cv < 0.10 else '需要更多样本'} |\n")

    report_text = "\n".join(report)
    with open(RESULTS_DIR / "ablation_stats.md", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)


if __name__ == "__main__":
    # 运行3次实验
    run_multiple_times(num_runs=3)