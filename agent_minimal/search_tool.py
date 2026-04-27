#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

VECTOR_DIR = "rag/vector_store"
INDEX_PATH = os.path.join(VECTOR_DIR, "faiss.index")
META_PATH = os.path.join(VECTOR_DIR, "chunks_meta.json")
EMB_MODEL = "/root/autodl-tmp/embedding_model/BAAI/bge-small-zh-v1___5"  # 若无本地模型，改为 "BAAI/bge-small-zh-v1.5"

class LawSearcher:
    def __init__(self, index_path=INDEX_PATH, meta_path=META_PATH, emb_model=EMB_MODEL):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"faiss index not found: {index_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta not found: {meta_path}")
        self.index = faiss.read_index(index_path)
        self.meta = json.load(open(meta_path, "r", encoding="utf-8"))
        self.emb = SentenceTransformer(emb_model)

    def search_law(self, query: str, top_k: int = 3):
        qv = self.emb.encode([query], normalize_embeddings=True)
        qv = np.array(qv, dtype=np.float32)
        scores, ids = self.index.search(qv, top_k)

        results = []
        for rank, (i, s) in enumerate(zip(ids[0], scores[0]), start=1):
            if i < 0 or i >= len(self.meta):
                continue
            item = self.meta[i]
            results.append({
                "rank": rank,
                "score": float(s),
                "source": item.get("source", ""),
                "title": item.get("title", ""),
                "content": item.get("content", "")[:500]
            })
        return results