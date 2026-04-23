# phase_a/evaluate_chunking.py
"""
切分策略的检索质量评测。

评测方法：
对每个测试案例的问题，用相同的Embedding模型检索
人工判断Top-3结果是否"真正相关"
计算每种切分策略的检索精准率

这是RAG系统中"检索质量"的核心评测方式，
工业上叫 Retrieval Precision@K
"""

import sys
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.append("..")

# 加载已有的Embedding模型（复用，不重新下载）
MODEL_CACHE = "/root/autodl-tmp/embedding_model"
CHUNK_DIR   = "./chunk_results"

# 测试案例：问题 + 期望检索到的法条关键词
# 这是"人工标注"的黄金标准
RETRIEVAL_GROUND_TRUTH = [
    # ====== 原有 6 个案例（保持不变） ======
    {
        "id": "TC001",
        "question": "我在公司工作了3年，公司突然裁员，能拿到什么赔偿？",
        "expected_keywords": ["第四十六条", "第四十七条", "经济补偿"],
    },
    {
        "id": "TC003",
        "question": "公司让我签竞业协议，但从来不给竞业补偿金，这个协议有效吗？",
        "expected_keywords": ["第二十三条", "第二十四条", "竞业"],
    },
    {
        "id": "TC004",
        "question": "我在上班途中骑车被撞伤了，算工伤吗？",
        "expected_keywords": ["第十四条", "上下班途中", "工伤"],
    },
    {
        "id": "TC007",
        "question": "朋友借了我5万块钱，已经2年了不还，现在还能起诉吗？",
        "expected_keywords": ["第一百八十八条", "诉讼时效", "三年"],
    },
    {
        "id": "TC009",
        "question": "同事在公司群里发消息说我贪污，完全是造谣，我能告他吗？",
        "expected_keywords": ["第一千零二十四条", "名誉权", "侵权"],
        "should_not_contain": ["劳动合同法"],
    },
    {
        "id": "TC010",
        "question": "我是外卖骑手，送餐时出了交通事故，平台说我是个人承包不是员工，不给报工伤",
        "expected_keywords": ["工伤", "劳动关系"],
    },
    
    # ====== 新增 10 个案例 ======
    {
        "id": "TC011",
        "question": "试用期被公司辞退，说我不符合录用条件，但我没看到过录用条件，能要赔偿吗？",
        "expected_keywords": ["第二十一条", "第三十九条", "试用期", "不符合录用条件"],
        "should_not_contain": ["第四十六条", "经济补偿"],  # 试用期无经济补偿
    },
    {
        "id": "TC012",
        "question": "公司拖欠工资3个月了，我如果直接走人不干了，能要求公司给补偿吗？",
        "expected_keywords": ["第三十八条", "被迫解除", "经济补偿", "工资"],
    },
    {
        "id": "TC013",
        "question": "入职时签了自愿放弃社保的协议，现在离职了还能要求公司补缴社保吗？",
        "expected_keywords": ["第七十二条", "社会保险", "强制性", "无效"],
        "should_not_contain": ["协商一致"],
    },
    {
        "id": "TC014",
        "question": "生病请了病假，公司说只发基本工资的60%，合法吗？",
        "expected_keywords": ["病假工资", "医疗期", "80%", "最低工资"],
    },
    {
        "id": "TC015",
        "question": "怀孕5个月了，公司以经营困难为由要裁掉我，我能不同意吗？",
        "expected_keywords": ["第四十二条", "女职工", "孕期", "不得解除"],
        "should_not_contain": ["经济性裁员"],
    },
    {
        "id": "TC016",
        "question": "劳务派遣员工在用工单位受伤，应该找派遣公司还是用工单位赔？",
        "expected_keywords": ["劳务派遣", "工伤", "用工单位", "连带责任"],
    },
    {
        "id": "TC017",
        "question": "公司说我是小时工，不给交社保也不签合同，合法吗？",
        "expected_keywords": ["非全日制用工", "小时工", "劳动合同", "社会保险"],
        "should_not_contain": ["全日制"],
    },
    {
        "id": "TC018",
        "question": "加班费应该怎么算？平时、周末和法定节假日有什么区别？",
        "expected_keywords": ["第四十四条", "加班费", "150%", "200%", "300%"],
    },
    {
        "id": "TC019",
        "question": "合同到期公司不续签了，能拿补偿金吗？",
        "expected_keywords": ["第四十六条", "劳动合同期满", "不续签", "经济补偿"],
    },
    {
        "id": "TC020",
        "question": "口头约定的提成奖金，离职时公司不承认怎么办？",
        "expected_keywords": ["奖金", "提成", "工资", "举证责任"],
    },
]


def build_temp_index(chunks: list[dict], model: SentenceTransformer):
    """为一批chunks构建临时FAISS索引"""
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return index


def evaluate_retrieval(
    chunks: list[dict],
    model: SentenceTransformer,
    ground_truth: list[dict],
    top_k: int = 3
) -> dict:
    """
    评测检索精准率。

    对每个问题：
    1. 检索top-k个chunk
    2. 检查是否包含expected_keywords（至少一个）
    3. 检查是否包含should_not_contain（噪声法条）
    计算 Precision@K 和 Noise Rate
    """
    index = build_temp_index(chunks, model)
    results = []

    for gt in ground_truth:
        query_vec = model.encode([gt["question"]], normalize_embeddings=True)
        scores, indices = index.search(query_vec.astype(np.float32), k=top_k)

        retrieved_texts = [chunks[i]["text"] for i in indices[0] if i < len(chunks)]
        retrieved_combined = " ".join(retrieved_texts)

        # 检查是否命中期望关键词（至少命中一个）
        hit_keywords = [
            kw for kw in gt["expected_keywords"]
            if kw in retrieved_combined
        ]
        is_relevant = len(hit_keywords) > 0

        # 检查是否有噪声（不应该出现的内容）
        noise_keywords = gt.get("should_not_contain", [])
        noise_hit = [kw for kw in noise_keywords if kw in retrieved_combined]
        has_noise = len(noise_hit) > 0

        results.append({
            "id":          gt["id"],
            "relevant":    is_relevant,
            "hit_keywords": hit_keywords,
            "has_noise":   has_noise,
            "noise_hit":   noise_hit,
            "top_sources": [chunks[i]["source"] for i in indices[0] if i < len(chunks)],
        })

        # 打印详情
        status = "✅" if is_relevant else "❌"
        noise_flag = " ⚠️噪声" if has_noise else ""
        print(f"  {status} {gt['id']}: 命中{hit_keywords}{noise_flag}")
        print(f"     来源: {[chunks[i]['source'] for i in indices[0]]}")

    precision = sum(r["relevant"] for r in results) / len(results)
    noise_rate = sum(r["has_noise"] for r in results) / len(results)

    return {
        "precision_at_k": precision,
        "noise_rate":     noise_rate,
        "details":        results,
    }


def load_model() -> SentenceTransformer:
    """加载已有的bge-small模型"""
    import os
    for root, dirs, files in os.walk(MODEL_CACHE):
        if "config.json" in files and "bge" in root.lower():
            print(f"⚡ 从缓存加载: {root}")
            return SentenceTransformer(root)
    return SentenceTransformer("BAAI/bge-small-zh-v1.5", cache_folder=MODEL_CACHE)


if __name__ == "__main__":
    import os

    print("=" * 60)
    print("切分策略检索质量评测（Precision@3）")
    print("=" * 60)

    model = load_model()
    all_eval_results = {}

    for strategy_file, strategy_name in [
        ("chunks_A.json", "A（按条款）"),
        ("chunks_B.json", "B（固定200字）"),
        ("chunks_C.json", "C（递归语义）"),
    ]:
        path = os.path.join(CHUNK_DIR, strategy_file)
        if not os.path.exists(path):
            print(f"⚠️  跳过 {strategy_name}（文件不存在）")
            continue

        with open(path) as f:
            chunks = json.load(f)

        print(f"\n{'='*40}")
        print(f"策略{strategy_name}（{len(chunks)}个chunks）")

        eval_result = evaluate_retrieval(chunks, model, RETRIEVAL_GROUND_TRUTH)
        all_eval_results[strategy_name] = eval_result

        print(f"  Precision@3: {eval_result['precision_at_k']:.1%}")
        print(f"  噪声率: {eval_result['noise_rate']:.1%}")

    # 汇总对比
    print("\n" + "=" * 60)
    print("切分策略评测汇总：")
    print(f"{'策略':<20} {'Precision@3':>12} {'噪声率':>8}")
    print("-" * 42)
    for name, result in all_eval_results.items():
        print(f"{name:<20} {result['precision_at_k']:>11.1%} {result['noise_rate']:>7.1%}")

    # 保存结果
    os.makedirs("./results", exist_ok=True)
    with open("./results/chunking_eval.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: {
                "precision_at_k": v["precision_at_k"],
                "noise_rate":     v["noise_rate"]
             } for k, v in all_eval_results.items()},
            f, ensure_ascii=False, indent=2
        )
    print("\n结果已保存到 ./results/chunking_eval.json")