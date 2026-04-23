# phase_c/enterprise_pipeline.py
"""
完整企业级RAG Pipeline

整合：
1. Phase A最优切分策略的chunks
2. Phase B选出的最优Embedding模型
3. Hybrid检索（BM25 + Dense + RRF）
4. BGE-Reranker精排
5. 引用溯源格式化输出
"""

import sys
import json
import time
from openai import OpenAI

sys.path.append("..")
from phase_c.hybrid_retriever   import HybridRetriever
from phase_c.reranker            import LawReranker
from phase_c.citation_formatter  import build_citation_prompt, format_final_output

VLLM_BASE_URL = "http://localhost:6006/v1"
MODEL_NAME    = "law-expert"
TEMPERATURE   = 0.1

BASE_SYSTEM_PROMPT = """你是一名专业的中国法律顾问，具有10年执业经验，专注于劳动法、合同法和民事纠纷领域。

## 回答规范
1. 引用法律条文时，必须包含完整名称和条款编号
   正确格式：《劳动合同法》第四十六条
2. 回答结构：【法律分析】→ 【具体建议】→ 【维权途径】
3. 涉及赔偿金额时，必须给出计算公式

## 禁止行为
- 禁止模糊表述（"相关法律规定"等）
- 禁止承诺诉讼结果"""


class EnterpriseLawRAG:
    """
    企业级法律RAG系统。
    
    支持离线和在线模式初始化。
    """
    def __init__(self, chunks: list[dict], offline_mode: bool = True):
        """
        初始化企业级RAG系统

        Args:
            chunks (list[dict]): Phase A最优切分的chunks数据
            offline_mode (bool): 是否使用离线模式
                - True: Reranker仅使用本地缓存（推荐在网络不稳定时使用）
                - False: Reranker允许从网络下载模型

        Raises:
            FileNotFoundError: 离线模式下找不到本地模型时抛出
        """
        print("初始化企业级RAG系统...")
        print(f"  - 离线模式: {offline_mode}")
        
        self.hybrid    = HybridRetriever(chunks)
        print("  ✅ Hybrid检索器加载完成")
        
        try:
            # ✅ 传递offline_mode给Reranker
            self.reranker  = LawReranker(offline_mode=offline_mode)
            print("  ✅ Reranker加载完成")
        except FileNotFoundError as e:
            print(f"  ❌ Reranker初始化失败")
            raise
        
        self.client    = OpenAI(api_key="EMPTY", base_url=VLLM_BASE_URL)
        print("✅ 企业级RAG系统初始化完成\n")

    def answer(
        self,
        question: str,
        top_k:    int = 3,
        verbose:  bool = True
    ) -> dict:
        """
        完整RAG问答流程

        Args:
            question (str): 用户提问
            top_k (int): 精排后返回的top-k结果数
            verbose (bool): 是否打印详细日志

        Returns:
            dict: 包含answer、latency_s、citations、citation_stats的输出
        """
        start = time.time()

        # Step 1: Hybrid检索（粗排，top-20候选）
        candidates = self.hybrid.retrieve(question, top_k=20, candidate_k=20)

        if verbose:
            print(f"\n{'='*60}")
            print(f"问题: {question}")
            both_paths = sum(1 for c in candidates if c.get("both_paths"))
            print(f"Hybrid检索: {len(candidates)}个候选"
                  f"（{both_paths}个同时命中两路）")

        # Step 2: Reranker精排（top-20 → top-k）
        reranked = self.reranker.rerank(question, candidates, top_k=top_k)

        if verbose:
            print(f"Reranker精排后top-{top_k}：")
            for r in reranked:
                print(f"  [{r['rerank_score']:.4f}] {r['source']} | {r['article']}")

        # Step 3: 构建带引用指令的Prompt
        augmented_prompt, citation_meta = build_citation_prompt(
            BASE_SYSTEM_PROMPT, reranked
        )

        # Step 4: LLM生成
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": augmented_prompt},
                {"role": "user",   "content": question}
            ],
            temperature=TEMPERATURE,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content

        # Step 5: 格式化结构化输出（含引用解析）
        output = format_final_output(
            question, answer, citation_meta,
            latency=time.time() - start
        )

        if verbose:
            print(f"\n回答（{output['latency_s']:.2f}s）：")
            print(answer[:500] + "..." if len(answer) > 500 else answer)
            print(f"\n引用统计：检索{output['citation_stats']['retrieved_count']}条 | "
                  f"模型引用{output['citation_stats']['cited_count']}条 | "
                  f"成功匹配{output['citation_stats']['matched_count']}条")

        return output


if __name__ == "__main__":
    with open("../phase_a/chunk_results/chunks_C.json") as f:
        chunks = json.load(f)

    # ✅ 使用离线模式初始化
    rag = EnterpriseLawRAG(chunks, offline_mode=True)

    # 测试关键问题
    test_questions = [
        "公司裁员3年工龄能拿多少赔偿？",
        "同事在公司群里造谣说我贪污，我能告他吗？",  # TC009，测试Hybrid修复效果
        "公司让我签竞业协议但不给补偿金，有效吗？",   # TC003，测试Reranker效果
    ]

    all_outputs = []
    for q in test_questions:
        output = rag.answer(q, verbose=True)
        all_outputs.append(output)
        print("-" * 60)

    # 保存结果
    import os
    os.makedirs("./results", exist_ok=True)
    with open("./results/enterprise_pipeline_test.json", "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)
    print("\n✅ 结果已保存 ./results/enterprise_pipeline_test.json")