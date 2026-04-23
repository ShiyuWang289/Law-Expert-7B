"""
Step 3: 检索器核心类

设计为可复用的类，供RAG Pipeline和评测脚本调用。
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def _find_local_model(cache_dir: str, keyword: str) -> str | None:
    """在缓存目录下递归查找包含keyword且含config.json的模型路径"""
    if not os.path.exists(cache_dir):
        return None
    for root, dirs, files in os.walk(cache_dir):
        if "config.json" in files and keyword in root.lower():
            return root
    return None


class LawRetriever:
    """
    法律条款检索器。

    使用方式：
        retriever = LawRetriever()
        results = retriever.retrieve("公司裁员能拿多少赔偿", top_k=3)
        context = retriever.format_context(results)
    """

    def __init__(
        self,
        vector_dir: str = "./vector_store",
        model_cache: str = "/root/autodl-tmp/embedding_model",
        model_name: str = "BAAI/bge-small-zh-v1.5",
    ):
        self.vector_dir = vector_dir
        self._load_index(vector_dir)
        self._load_model(model_name, model_cache)

    def _load_index(self, vector_dir: str):
        """加载FAISS索引和chunk元数据"""
        index_path = os.path.join(vector_dir, "faiss.index")
        meta_path  = os.path.join(vector_dir, "chunks_meta.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"找不到索引文件: {index_path}\n"
                "请先运行 02_build_index.py"
            )

        self.index  = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        print(f"✅ 索引加载完成: {self.index.ntotal} 条向量")

    def _load_model(self, model_name: str, model_cache: str):
        """加载Embedding模型，优先使用本地缓存"""
        # 优先从本地缓存查找
        local_path = _find_local_model(model_cache, "bge-small-zh")
        if local_path:
            print(f"⚡ 从本地缓存加载模型: {local_path}")
            self.model = SentenceTransformer(local_path)
        else:
            print(f"⬇️  从网络加载模型: {model_name}")
            self.model = SentenceTransformer(
                model_name,
                cache_folder=model_cache
            )
        print(f"✅ Embedding模型加载完成")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.3
    ) -> list[dict]:
        """
        检索最相关的法律条款。

        Args:
            query: 用户问题
            top_k: 返回条款数量
            score_threshold: 相似度阈值，低于此值的结果不返回

        Returns:
            list of {text, source, article, score}
        """
        # 向量化查询
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        # FAISS检索
        scores, indices = self.index.search(
            query_vec.astype(np.float32),
            k=top_k
        )

        # 组装结果，过滤低分
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score < score_threshold:
                continue
            result = dict(self.chunks[idx])  # 复制，避免修改原数据
            result["score"] = float(score)
            results.append(result)

        return results

    def format_context(self, results: list[dict]) -> str:
        """
        将检索结果格式化为注入Prompt的上下文字符串。
        """
        if not results:
            return ""
    
        lines = ["【参考法条】（以下法条由系统检索提供，请优先参考）\n"]
        for i, result in enumerate(results):
            # 兼容 'content' 和 'text' 两种字段名
            text = result.get('content', result.get('text', ''))
            lines.append(f"{i+1}. {text}\n")
    
        return "\n".join(lines)

    def build_rag_prompt(
        self,
        question: str,
        system_prompt: str,
        top_k: int = 3
    ) -> tuple[str, list[dict]]:
        """
        完整的RAG Prompt构建。

        Returns:
            (augmented_system_prompt, retrieved_results)
        """
        results = self.retrieve(question, top_k=top_k)
        context = self.format_context(results)

        if context:
            augmented_prompt = f"{system_prompt}\n\n{context}"
        else:
            augmented_prompt = system_prompt

        return augmented_prompt, results