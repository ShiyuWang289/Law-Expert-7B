# phase_c/hybrid_retriever.py
"""
Hybrid 检索器：BM25（稀疏）+ Dense（稠密）+ RRF融合

RRF（Reciprocal Rank Fusion）算法：
score(d) = Σ 1/(k + rank_i(d))
k=60（标准参数，平滑排名差异）

为什么用RRF而不是线性加权：
- 线性加权需要调参（权重比例）
- RRF无需调参，对排名敏感而非原始分数
- 在两路中都排前的文档自然得高分
- 工业界标准做法，经过大量验证
"""

import numpy as np
import faiss
import json
import sys
import os
from sentence_transformers import SentenceTransformer

sys.path.append("..")
from phase_c.bm25_index import load_bm25, bm25_search, build_bm25_index, save_bm25

MODEL_CACHE = "/root/autodl-tmp/embedding_model"
BM25_PATH   = "../vector_store/bm25_index.pkl"


def find_best_model_path() -> str:
    for keyword in ["bge-base-zh", "bge-small-zh"]:
        for root, dirs, files in os.walk(MODEL_CACHE):
            if "config.json" in files and keyword in root.lower():
                return root
    raise FileNotFoundError("找不到Embedding模型")


def rrf_fusion(
    dense_results:  list[dict],
    sparse_results: list[dict],
    k: int = 60
) -> list[dict]:
    """
    RRF算法融合两路检索结果。

    Args:
        dense_results:  Dense检索结果（已按相关度排序）
        sparse_results: BM25检索结果（已按相关度排序）
        k: RRF平滑参数（默认60）

    Returns:
        融合后的排序结果（按RRF分数降序）
    """
    # 用(source, article)作为文档唯一标识
    doc_scores: dict[str, float] = {}
    doc_data:   dict[str, dict]  = {}

    # Dense路的RRF分数
    for rank, doc in enumerate(dense_results, start=1):
        doc_id = f"{doc['source']}_{doc['article']}"
        rrf_score = 1.0 / (k + rank)
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
        if doc_id not in doc_data:
            doc_data[doc_id] = dict(doc)
            doc_data[doc_id]["in_dense"]  = True
            doc_data[doc_id]["in_sparse"] = False
        doc_data[doc_id]["dense_rank"] = rank

    # Sparse路的RRF分数
    for rank, doc in enumerate(sparse_results, start=1):
        doc_id = f"{doc['source']}_{doc['article']}"
        rrf_score = 1.0 / (k + rank)
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
        if doc_id not in doc_data:
            doc_data[doc_id] = dict(doc)
            doc_data[doc_id]["in_dense"]  = False
            doc_data[doc_id]["in_sparse"] = True
        else:
            doc_data[doc_id]["in_sparse"] = True
        doc_data[doc_id]["sparse_rank"] = rank

    # 按RRF分数排序
    sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

    results = []
    for doc_id in sorted_ids:
        doc = doc_data[doc_id]
        doc["rrf_score"]  = doc_scores[doc_id]
        doc["both_paths"] = doc.get("in_dense", False) and doc.get("in_sparse", False)
        results.append(doc)

    return results


class HybridRetriever:
    """
    Hybrid检索器：Dense + BM25 + RRF融合

    使用方式：
        retriever = HybridRetriever(chunks)
        results = retriever.retrieve("公司裁员赔偿", top_k=3)
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self._build_dense_index()
        self._build_sparse_index()

    def _build_dense_index(self):
        """构建Dense向量索引"""
        model_path  = find_best_model_path()
        self.model  = SentenceTransformer(model_path)
        texts       = [c["text"] for c in self.chunks]
        embeddings  = self.model.encode(
            texts, normalize_embeddings=True,
            show_progress_bar=True, batch_size=32
        )
        self.dense_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.dense_index.add(embeddings.astype(np.float32))
        print(f"✅ Dense索引: {self.dense_index.ntotal}条向量")

    def _build_sparse_index(self):
        """构建BM25稀疏索引"""
        if os.path.exists(BM25_PATH):
            self.bm25, _ = load_bm25(BM25_PATH)
        else:
            self.bm25 = build_bm25_index(self.chunks)
            save_bm25(self.bm25, self.chunks)

    def retrieve(
        self,
        query:         str,
        top_k:         int = 3,
        candidate_k:   int = 20,
    ) -> list[dict]:
        """
        Hybrid检索主流程：
        1. Dense检索 top-20候选
        2. BM25检索 top-20候选
        3. RRF融合
        4. 返回top_k结果
        """
        # Dense检索
        query_vec = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.dense_index.search(
            query_vec.astype(np.float32), k=candidate_k
        )
        dense_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                result = dict(self.chunks[idx])
                result["dense_score"] = float(score)
                dense_results.append(result)

        # BM25检索
        sparse_results = bm25_search(
            self.bm25, self.chunks, query, top_k=candidate_k
        )

        # RRF融合
        fused = rrf_fusion(dense_results, sparse_results)

        return fused[:top_k]