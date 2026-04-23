# phase_b/embedding_benchmark.py
"""
Embedding模型选型实验

对比：
- bge-small-zh-v1.5（当前，~92MB）
- bge-base-zh-v1.5（升级，~400MB）
- m3e-base（横向对比，~400MB）

评测维度：
1. 向量化速度（条/秒）
2. 显存/内存占用
3. Precision@3（复用Phase A的评测逻辑）
4. 相似度分布（区分度是否够好）

固定条件：使用Phase A选出的最优切分策略
"""

import os
import sys
import time
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../phase_a"))
from evaluate_chunking import evaluate_retrieval, RETRIEVAL_GROUND_TRUTH

MODEL_CACHE = "/root/autodl-tmp/embedding_model"

# 三个候选模型的本地路径查找逻辑
MODEL_CONFIGS = {
    "bge-small-zh": {
        "search_keyword": "bge-small-zh",
        "hf_name":        "BAAI/bge-small-zh-v1.5",
        "expected_dim":   512,
    },
    "bge-base-zh": {
        "search_keyword": "bge-base-zh",
        "hf_name":        "AI-ModelScope/bge-base-zh-v1.5",
        "expected_dim":   768,
    },
}


def find_local_model(search_keyword: str) -> str | None:
    """在embedding_model目录下查找已下载的模型"""
    for root, dirs, files in os.walk(MODEL_CACHE):
        if "config.json" in files and search_keyword.lower() in root.lower():
            return root
    return None


def load_model(config: dict) -> SentenceTransformer:
    """加载模型，优先本地缓存"""
    local_path = find_local_model(config["search_keyword"])
    if local_path:
        print(f"⚡ 从缓存加载: {local_path}")
        return SentenceTransformer(local_path)
    else:
        print(f"⬇️  从网络下载: {config['hf_name']}")
        return SentenceTransformer(config["hf_name"], cache_folder=MODEL_CACHE)


def benchmark_model(
    model_name: str,
    model: SentenceTransformer,
    chunks: list[dict]
) -> dict:
    """对单个Embedding模型进行完整benchmark"""
    print(f"\n{'='*50}")
    print(f"Benchmark: {model_name}")
    print(f"{'='*50}")

    texts = [c["text"] for c in chunks]

    # 1. 速度测试
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    speed_elapsed = time.time() - start
    speed = len(texts) / speed_elapsed

    print(f"向量维度: {embeddings.shape[1]}")
    print(f"向量化速度: {speed:.1f} 条/秒")
    print(f"总耗时: {speed_elapsed:.1f}s")

    # 2. 构建临时FAISS索引
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))

    # 3. 相似度分布分析（判断区分度）
    # 随机采样10对query-doc，看相似度分布是否有区分度
    sample_queries = [gt["question"] for gt in RETRIEVAL_GROUND_TRUTH[:6]]
    query_vecs = model.encode(sample_queries, normalize_embeddings=True)
    all_scores, _ = index.search(query_vecs.astype(np.float32), k=len(chunks))
    top1_scores  = all_scores[:, 0]
    top3_scores  = all_scores[:, 2]
    score_gap    = (top1_scores - top3_scores).mean()  # top1和top3的分数差

    print(f"Top-1平均相似度: {top1_scores.mean():.4f}")
    print(f"Top-3平均相似度: {top3_scores.mean():.4f}")
    print(f"区分度gap(top1-top3): {score_gap:.4f}（越大越好）")

    # 4. Precision@3评测（复用Phase A逻辑）
    print(f"\n检索质量评测（Precision@3）：")
    eval_result = evaluate_retrieval(chunks, model, RETRIEVAL_GROUND_TRUTH)

    return {
        "model_name":    model_name,
        "vector_dim":    int(embeddings.shape[1]),
        "speed_per_sec": round(speed, 1),
        "total_time_s":  round(speed_elapsed, 1),
        "top1_score":    round(float(top1_scores.mean()), 4),
        "score_gap":     round(float(score_gap), 4),
        "precision_at_3": eval_result["precision_at_k"],
        "noise_rate":    eval_result["noise_rate"],
    }


if __name__ == "__main__":
    # 加载Phase A选出的最优切分结果（根据Phase A实际结果修改文件名）
    chunk_file = "../phase_a/chunk_results/chunks_C.json"  # 默认用策略C
    with open(chunk_file) as f:
        chunks = json.load(f)
    print(f"使用切分策略C，共{len(chunks)}个chunks")

    all_results = []

    for model_name, config in MODEL_CONFIGS.items():
        try:
            model  = load_model(config)
            result = benchmark_model(model_name, model, chunks)
            all_results.append(result)
            del model  # 释放内存
        except Exception as e:
            print(f"❌ {model_name} 加载失败: {e}")

    # 汇总对比表
    print("\n" + "=" * 70)
    print("Embedding模型选型汇总：")
    print(f"{'模型':<20} {'维度':>6} {'速度':>8} {'精准率':>8} {'噪声率':>8} {'区分度':>8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['model_name']:<20} {r['vector_dim']:>6} "
              f"{r['speed_per_sec']:>7.1f}/s "
              f"{r['precision_at_3']:>7.1%} "
              f"{r['noise_rate']:>7.1%} "
              f"{r['score_gap']:>8.4f}")

    os.makedirs("./results", exist_ok=True)
    with open("./results/embedding_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\n结果已保存 ./results/embedding_benchmark.json")