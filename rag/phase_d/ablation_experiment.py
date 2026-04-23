# phase_d/ablation_experiment.py
"""
五版本消融实验（修复版）

核心修复：
1. V2 retriever.py 使用本地模型路径（不依赖HuggingFace网络）
2. V3 = bge-base + 纯Dense，不含Reranker（与V4形成真正对比）
3. V4 = 完整Hybrid + Reranker（才是真正的"新增组件"）
4. V5 = V4 + 统计引用率
"""

import sys, os, json, time
from pathlib import Path
from openai import OpenAI

PHASE_D_DIR = Path(__file__).parent.absolute()
RAG_DIR     = PHASE_D_DIR.parent
LLAMA_ROOT  = RAG_DIR.parent

sys.path.insert(0, str(LLAMA_ROOT))
sys.path.insert(0, str(RAG_DIR))

from prompt_engineering.evaluator   import evaluate_single_response
from prompt_engineering.test_cases  import TEST_CASES
from phase_c.enterprise_pipeline    import EnterpriseLawRAG, BASE_SYSTEM_PROMPT
from phase_c.citation_formatter     import build_citation_prompt
from retriever                      import LawRetriever
from rag_pipeline                   import ask_without_rag, ask_with_rag

RESULTS_DIR   = PHASE_D_DIR / "results"
VLLM_BASE_URL = "http://localhost:6006/v1"
MODEL_NAME    = "law-expert"
TEMPERATURE   = 0.1


# ============================================================
# V3专用：纯Dense检索（不含BM25和Reranker，体现切分+模型的贡献）
# ============================================================
def answer_v3_dense_only(question: str, hybrid_retriever, client: OpenAI) -> dict:
    """
    V3：bge-base + 纯Dense（FAISS）检索，无BM25，无Reranker
    目的：单独验证"更好的Embedding + 更优切分"的贡献
    """
    import numpy as np

    start = time.time()

    # 只用Dense，不走BM25也不走Reranker
    qv = hybrid_retriever.model.encode([question], normalize_embeddings=True)
    scores, indices = hybrid_retriever.dense_index.search(
        qv.astype('float32'), k=3
    )
    chunks = hybrid_retriever.chunks
    top_chunks = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks):
            c = dict(chunks[idx])
            c['rerank_score'] = float(score)
            top_chunks.append(c)

    # 用基础Prompt（不含引用指令，和V2保持一致的Prompt复杂度）
    context = "\n\n".join([
        f"[参考{i+1}] {c['source']} {c['article']}\n{c['text']}"
        for i, c in enumerate(top_chunks)
    ])
    system = f"{BASE_SYSTEM_PROMPT}\n\n【参考法条】\n{context}"

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": question}],
        temperature=TEMPERATURE, max_tokens=1024
    )
    return {
        "question":  question,
        "answer":    resp.choices[0].message.content,
        "latency":   time.time() - start,
        "rag_used":  True,
        "retrieval": [f"{c['source']}|{c['article']}" for c in top_chunks],
    }


# ============================================================
# V4/V5：完整企业级Pipeline（Hybrid + Reranker）
# ============================================================
def answer_v4_enterprise(question: str, rag: EnterpriseLawRAG,
                         with_citation_stats: bool = False) -> dict:
    """V4/V5 共用：完整Hybrid + Reranker Pipeline"""
    output = rag.answer(question, top_k=3, verbose=False)
    result = {
        "question": question,
        "answer":   output["answer"],
        "latency":  output["latency_s"],
        "rag_used": True,
        "retrieval": [c.get('source_id','') for c in output.get('citations', [])],
    }
    if with_citation_stats:
        result["citation_stats"] = output.get("citation_stats", {})
    return result


# ============================================================
# 主实验流程
# ============================================================
def run_ablation():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key="EMPTY", base_url=VLLM_BASE_URL)

    # 加载chunks（Phase A最优切分）
    with open(RAG_DIR / "phase_a/chunk_results/chunks_C.json") as f:
        chunks_c = json.load(f)
    print(f"chunks_C 共 {len(chunks_c)} 条")

    os.chdir(RAG_DIR)
    # 初始化各版本所需的检索器
    print("\n初始化 V2 检索器（bge-small + FAISS）...")
    try:
        simple_retriever = LawRetriever()
        v2_available = True
        print("✅ V2 检索器就绪")
    except Exception as e:
        print(f"❌ V2 初始化失败：{e}")
        v2_available = False

    print("\n初始化 V3/V4/V5 企业级RAG（bge-base + Hybrid + Reranker）...")
    enterprise_rag = EnterpriseLawRAG(chunks_c)
    print("✅ 企业级RAG就绪")

    versions = ["V1_no_rag", "V2_simple_rag", "V3_dense_bgebase",
                "V4_hybrid_reranker", "V5_full_citation"]

    all_results = []

    for version in versions:
        print(f"\n{'='*60}")
        print(f"测试版本: {version}")
        print(f"{'='*60}")

        if version == "V2_simple_rag" and not v2_available:
            print("⏭️  跳过（检索器初始化失败）")
            continue

        for tc in TEST_CASES:
            try:
                if version == "V1_no_rag":
                    raw = ask_without_rag(tc["question"], client)

                elif version == "V2_simple_rag":
                    raw = ask_with_rag(tc["question"], simple_retriever,
                                      client, top_k=3, verbose=False)

                elif version == "V3_dense_bgebase":
                    # ← 关键区别：只用Dense，无BM25无Reranker
                    raw = answer_v3_dense_only(
                        tc["question"], enterprise_rag.hybrid, client
                    )

                elif version == "V4_hybrid_reranker":
                    # ← 完整Pipeline
                    raw = answer_v4_enterprise(tc["question"], enterprise_rag,
                                               with_citation_stats=False)

                elif version == "V5_full_citation":
                    # ← 同V4，额外记录引用率
                    raw = answer_v4_enterprise(tc["question"], enterprise_rag,
                                               with_citation_stats=True)

                eval_result = evaluate_single_response(
                    raw["answer"], tc, version
                )
                eval_result["latency"]  = raw.get("latency", 0)
                eval_result["rag_used"] = raw.get("rag_used", False)
                eval_result["retrieval"] = raw.get("retrieval", [])

                # V5额外记录引用率
                if version == "V5_full_citation":
                    cs = raw.get("citation_stats", {})
                    eval_result["citation_rate"] = cs.get("citation_rate", 0)

                all_results.append(eval_result)

                score   = eval_result["scores"]["total"]
                kp_rate = eval_result["details"]["key_points"]["coverage_rate"]
                print(f"  {tc['id']} | 总分:{score:.1f} | 覆盖:{kp_rate:.0%} | {raw.get('latency',0):.1f}s")

            except Exception as e:
                import traceback
                print(f"  {tc['id']} ❌ 报错: {e}")
                traceback.print_exc()

    # 保存原始数据
    out_path = RESULTS_DIR / "ablation_raw.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 原始结果已保存: {out_path}")

    return all_results


# ============================================================
# 报告生成
# ============================================================
def generate_report(results: list) -> str:
    def avg(lst): return sum(lst) / len(lst) if lst else 0

    versions = ["V1_no_rag", "V2_simple_rag", "V3_dense_bgebase",
                "V4_hybrid_reranker", "V5_full_citation"]

    desc = {
        "V1_no_rag":           "纯微调，无RAG（基线）",
        "V2_simple_rag":       "简单RAG（bge-small+FAISS）",
        "V3_dense_bgebase":    "优化切分+bge-base（纯Dense）",
        "V4_hybrid_reranker":  "V3+BM25+RRF+Reranker",
        "V5_full_citation":    "V4+引用溯源（完整版）",
    }

    lines = ["# RAG系统消融实验报告\n", "## 总体对比\n",
             "| 版本 | 说明 | 平均总分 | 关键点覆盖 | 延迟 |",
             "|:---|:---|:---:|:---:|:---:|"]

    for v in versions:
        vr = [r for r in results if r.get("prompt_version") == v]
        if not vr:
            lines.append(f"| {v} | {desc.get(v,v)} | N/A | N/A | N/A |")
            continue
        avg_score = avg([r["scores"]["total"] for r in vr])
        avg_kp    = avg([r["details"]["key_points"]["coverage_rate"] for r in vr])
        avg_lat   = avg([r.get("latency", 0) for r in vr])
        lines.append(
            f"| {v} | {desc.get(v,v)} | {avg_score:.1f} | {avg_kp:.1%} | {avg_lat:.1f}s |"
        )

    # 版本间对比（关键结论）
    lines += ["\n## 组件贡献分析\n",
              "| 对比 | 新增组件 | 分数变化 | 结论 |",
              "|:---|:---|:---:|:---|"]

    def get_avg_score(v):
        vr = [r for r in results if r.get("prompt_version") == v]
        return avg([r["scores"]["total"] for r in vr]) if vr else None

    comparisons = [
        ("V1_no_rag",        "V2_simple_rag",      "RAG检索"),
        ("V2_simple_rag",    "V3_dense_bgebase",   "bge-base+优化切分"),
        ("V3_dense_bgebase", "V4_hybrid_reranker", "BM25+RRF+Reranker"),
        ("V4_hybrid_reranker","V5_full_citation",  "引用溯源"),
    ]
    for base_v, new_v, component in comparisons:
        base_score = get_avg_score(base_v)
        new_score  = get_avg_score(new_v)
        if base_score is not None and new_score is not None:
            delta = new_score - base_score
            symbol = "↑" if delta > 0.5 else ("↓" if delta < -0.5 else "≈")
            conclusion = "有效" if delta > 0.5 else ("有害" if delta < -0.5 else "持平")
            lines.append(
                f"| {base_v}→{new_v} | {component} | {delta:+.1f} {symbol} | {conclusion} |"
            )

    # V5引用率
    v5r = [r for r in results if r.get("prompt_version") == "V5_full_citation"]
    if v5r and any("citation_rate" in r for r in v5r):
        avg_cite = avg([r.get("citation_rate", 0) for r in v5r])
        lines += [f"\n## 引用溯源统计（V5）\n",
                  f"- 平均引用匹配率：**{avg_cite:.1%}**",
                  f"- 目标：>60%"]

    return "\n".join(lines)


if __name__ == "__main__":
    results = run_ablation()

    report = generate_report(results)
    report_path = RESULTS_DIR / "ablation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ 报告已生成: {report_path}")
    print("\n" + "="*60)
    print(report)