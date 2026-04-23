# phase_c/bm25_index.py
"""
BM25 稀疏检索索引

BM25 vs Dense Embedding 的本质区别：

Dense（向量）：
  "公司群里发消息" → 语义空间 → 可能找到"劳动合同"相关（语义相近）
  优点：能做语义泛化，找到同义词
  缺点：词汇陷阱（TC009问题）

BM25（词频统计）：
  "贪污" + "名誉" → 精确匹配 → 找到包含这些词的文档
  优点：精确匹配，不会被无关词汇带偏
  缺点：无法处理同义词（"赔偿"不能匹配"补偿"）

Hybrid = 两者取长补短
"""

import json
import jieba
import pickle
import os
from rank_bm25 import BM25Okapi

BM25_PATH = "../vector_store/bm25_index.pkl"


def tokenize_chinese(text: str) -> list[str]:
    """
    中文分词（BM25需要分词后的token列表）

    使用jieba分词，过滤停用词和单字
    停用词表：简版，保留法律术语
    """
    stopwords = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
        "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
        "你", "会", "着", "没有", "看", "好", "自己", "这", "那",
        "但", "如果", "因为", "所以", "可以", "应当", "应该",
    }

    words = jieba.cut(text, cut_all=False)
    tokens = [
        w.strip() for w in words
        if len(w.strip()) >= 2  # 过滤单字
        and w.strip() not in stopwords
    ]
    return tokens


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """构建BM25索引"""
    print("正在构建BM25索引...")
    tokenized_corpus = [tokenize_chinese(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"✅ BM25索引构建完成，共{len(chunks)}个文档")
    return bm25


def save_bm25(bm25: BM25Okapi, chunks: list[dict], path: str = BM25_PATH):
    """保存BM25索引（包含chunks元数据），自动创建目录"""
    # 将路径转为绝对路径，并确保目录存在
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    size = os.path.getsize(abs_path) / 1024
    print(f"BM25索引已保存: {abs_path} ({size:.1f}KB)")


def load_bm25(path: str = BM25_PATH) -> tuple[BM25Okapi, list[dict]]:
    """加载BM25索引"""
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"✅ BM25索引加载完成，{len(data['chunks'])}个文档")
    return data["bm25"], data["chunks"]


def bm25_search(
    bm25: BM25Okapi,
    chunks: list[dict],
    query: str,
    top_k: int = 10
) -> list[dict]:
    """
    BM25检索，返回带分数的结果列表。
    top_k设大一点（默认10），供后续RRF融合使用。
    """
    query_tokens = tokenize_chinese(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    # 取top_k
    top_indices = scores.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # 过滤零分文档
            result = dict(chunks[idx])
            result["bm25_score"] = float(scores[idx])
            result["bm25_rank"]  = len(results) + 1
            results.append(result)

    return results


if __name__ == "__main__":
    # 测试BM25
    with open("../phase_a/chunk_results/chunks_C.json") as f:
        chunks = json.load(f)

    bm25 = build_bm25_index(chunks)
    save_bm25(bm25, chunks)

    # 验证：TC009的BM25检索结果
    query = "同事在公司群里发消息说我贪污，完全是造谣，我能告他吗？"
    results = bm25_search(bm25, chunks, query, top_k=5)

    print(f"\nBM25检索结果（'{query[:20]}...'）：")
    for r in results:
        print(f"  [{r['bm25_score']:.3f}] {r['source']} | {r['article']}")
        print(f"    {r['text'][:60]}...")