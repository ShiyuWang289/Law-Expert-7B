"""
Step 4: RAG完整Pipeline

整合检索器和vLLM推理服务，提供完整的RAG问答能力。
"""

import time
from openai import OpenAI
from retriever import LawRetriever

# ============================================================
# 配置
# ============================================================
VLLM_BASE_URL = "http://localhost:6006/v1"
MODEL_NAME    = "law-expert"
TEMPERATURE   = 0.1

# 使用实验中效果最好的 v2_structured prompt
BASE_SYSTEM_PROMPT = """你是一名专业的中国法律顾问，具有10年执业经验，专注于劳动法、合同法和民事纠纷领域。

## 回答规范
1. 引用法律条文时，必须包含完整名称和条款编号
   正确格式：《劳动合同法》第四十六条
2. 回答结构：【法律分析】→ 【具体建议】→ 【维权途径】
3. 涉及赔偿金额时，必须给出计算公式

## 禁止行为
- 禁止模糊表述（"相关法律规定"等）
- 禁止承诺诉讼结果"""


def ask_with_rag(
    question: str,
    retriever: LawRetriever,
    client: OpenAI,
    top_k: int = 3,
    verbose: bool = True
) -> dict:
    """
    带RAG的完整问答流程。

    Returns:
        {question, answer, retrieved_laws, latency, rag_used}
    """
    start = time.time()

    # Step 1: 检索相关法条
    augmented_prompt, retrieved = retriever.build_rag_prompt(
        question,
        BASE_SYSTEM_PROMPT,
        top_k=top_k
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"问题: {question}")
        print(f"{'='*60}")
        if retrieved:
            print(f"\n📚 检索到 {len(retrieved)} 条相关法条:")
            for r in retrieved:
                print(f"  [{r['score']:.3f}] {r['article']}")
        else:
            print("⚠️  未检索到相关法条（将使用模型自身知识）")

    # Step 2: 调用vLLM生成回答
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": augmented_prompt},
            {"role": "user",   "content": question}
        ],
        temperature=TEMPERATURE,
        max_tokens=1024,
    )

    answer  = response.choices[0].message.content
    latency = time.time() - start

    if verbose:
        print(f"\n💬 回答 (耗时 {latency:.1f}s):")
        print(answer)

    return {
        "question":       question,
        "answer":         answer,
        "retrieved_laws": [r["article"] for r in retrieved],
        "retrieved_scores": [r["score"] for r in retrieved],
        "latency":        latency,
        "rag_used":       len(retrieved) > 0,
        "prompt_length":  len(augmented_prompt)
    }


def ask_without_rag(
    question: str,
    client: OpenAI,
    verbose: bool = False
) -> dict:
    """不带RAG的问答（用于对比）"""
    start = time.time()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user",   "content": question}
        ],
        temperature=TEMPERATURE,
        max_tokens=1024,
    )

    answer  = response.choices[0].message.content
    latency = time.time() - start

    if verbose:
        print(f"\n💬 无RAG回答 (耗时 {latency:.1f}s):")
        print(answer)

    return {
        "question": question,
        "answer":   answer,
        "latency":  latency,
        "rag_used": False
    }


if __name__ == "__main__":
    # 初始化
    retriever = LawRetriever()
    client    = OpenAI(api_key="EMPTY", base_url=VLLM_BASE_URL)

    # 测试问题（选择能体现RAG价值的问题）
    test_questions = [
        "我在公司工作了3年，公司突然裁员，能拿到什么赔偿？",
        "公司让我签竞业协议，但从来不给竞业补偿金，这个协议有效吗？",
        "我在上班途中骑车被撞伤了，算工伤吗？",
        "朋友借了我5万块钱，已经2年了不还，现在还能起诉吗？",
    ]

    print("🚀 RAG Pipeline 测试开始")
    print(f"模型: {MODEL_NAME} | 检索top_k: 3\n")

    for q in test_questions:
        result = ask_with_rag(q, retriever, client, top_k=3, verbose=True)
        print(f"\n检索法条: {result['retrieved_laws']}")
        print(f"Prompt长度: {result['prompt_length']} 字符")
        print("-" * 60)
