"""
BGE Reranker - 简洁可控的Cross-encoder实现

目标：
1. 加载预训练的cross-encoder模型
2. 对候选文档进行相关性打分
3. 按分数排序返回top_k
4. 分数应在合理的0-1范围内
"""

import os
from sentence_transformers import CrossEncoder

MODEL_CACHE = "/root/autodl-tmp/embedding_model"


def find_reranker_path():
    """查找本地cross-encoder模型"""
    candidates = [
        os.path.join(MODEL_CACHE, "bge-reranker-base"),
        os.path.join(MODEL_CACHE, "BAAI/bge-reranker-base"),
    ]
    
    for path in candidates:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
            return path
    
    # 如果都不存在，扫描整个缓存目录
    for root, dirs, files in os.walk(MODEL_CACHE):
        if "config.json" in files and "reranker" in root.lower():
            return root
    
    return None


class LawReranker:
    """法律场景Cross-encoder Reranker"""

    def __init__(self, offline_mode=True):
        """
        初始化Reranker
        
        Args:
            offline_mode: 离线模式（本地模型）
        """
        print("初始化Reranker...")
        
        model_path = find_reranker_path()
        if not model_path:
            raise FileNotFoundError(
                "❌ 找不到bge-reranker-base模型\n"
                f"应该位于: {MODEL_CACHE}"
            )
        
        print(f"  ⚡ 加载Cross-encoder: {model_path}")
        self.model = CrossEncoder(model_path, device="cpu")
        print(f"  ✅ Reranker加载完成")

    def rerank(self, query, candidates, top_k=3, threshold=0.0):
        """
        对候选文档进行相关性排序
        
        Args:
            query (str): 用户查询
            candidates (list[dict]): 候选文档，每个包含 'text' 和 'source' 字段
            top_k (int): 返回的排序结果数
            threshold (float): 分数阈值，低于此值过滤（0表示不过滤）
        
        Returns:
            list[dict]: 排序后的候选文档，新增 'rerank_score' 字段
        """
        if not candidates:
            return []
        
        # 构建 (query, doc) 对
        pairs = [(query, c["text"]) for c in candidates]
        
        # 使用cross-encoder计算相关性分数
        scores = self.model.predict(pairs)
        
        # 将候选和分数配对，然后按分数排序（降序）
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 选择top_k并过滤
        results = []
        for candidate, score in ranked[:top_k]:
            if threshold > 0 and score < threshold:
                continue
            
            result = dict(candidate)
            result["rerank_score"] = float(score)
            results.append(result)
        
        return results

