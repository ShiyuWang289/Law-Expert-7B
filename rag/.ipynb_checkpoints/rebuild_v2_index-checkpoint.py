# rag/rebuild_v2_index.py
"""
重建V2版本的FAISS索引，使用与V3/V4/V5相同的chunks_C.json
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MODEL_CACHE = "/root/autodl-tmp/embedding_model"
CHUNKS_PATH = "./phase_a/chunk_results/chunks_C.json"  # ✅ 与V3/V4/V5相同
OUTPUT_DIR = "./vector_store"

def main():
    print("🚀 重建V2版本的FAISS索引（使���chunks_C.json）...\n")
    
    # 加载chunks
    print(f"📂 加载chunks: {CHUNKS_PATH}")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"✅ 加载{len(chunks)}个chunks\n")
    
    # 加载模型
    print(f"🤖 加载embedding模型: {MODEL_NAME}")
    local_dirs = [d for d in os.listdir(MODEL_CACHE) if "bge-small-zh" in d]
    if local_dirs:
        local_path = os.path.join(MODEL_CACHE, local_dirs[0])
        for root, dirs, files in os.walk(local_path):
            if "config.json" in files:
                local_path = root
                break
        print(f"⚡ 从本地缓存加载: {local_path}")
        model = SentenceTransformer(local_path)
    else:
        model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE)
    print("✅ 模型加载完成\n")
    
    # 编码chunks
    print("🔄 编码chunks...")
    texts = [chunk.get("text", "") for chunk in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    )
    embeddings = embeddings.astype(np.float32)
    print(f"✅ 生成{len(embeddings)}个向量\n")
    
    # 构建FAISS索引
    print("🔨 构建FAISS索引...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"✅ FAISS索引构建完成\n")
    
    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    index_path = os.path.join(OUTPUT_DIR, "faiss.index")
    faiss.write_index(index, index_path)
    
    meta_path = os.path.join(OUTPUT_DIR, "chunks_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 索引已保存")
    print(f"   向量数: {index.ntotal}")
    print(f"   索引文件: {index_path}")
    print(f"   元数据: {meta_path}")

if __name__ == "__main__":
    main()