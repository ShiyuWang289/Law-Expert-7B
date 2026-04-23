"""
Step 2: 构建FAISS向量索引

关键设计决策：
1. Embedding模型选择 BAAI/bge-small-zh-v1.5
   - 模型大小：~100MB（V100磁盘和显存都友好）
   - 中文效果：中文MTEB榜单小模型第一
   - 速度：CPU上约500条/秒，够用
   - 替代方案：bge-base-zh（更准但更慢，~400MB）

2. 文本切分策略：按条款切分（而非固定字数）
   - 每个《法律》第X条 作为一个独立chunk
   - 好处：检索精度高，不会把不同条款混在一起
   - 坏处：条款太长时可能超过模型最大输入长度（512 token）

3. FAISS索引类型：IndexFlatIP（内积 = 余弦相似度）
   - 适合小规模语料（<10万条）
   - 精确检索，无近似误差
   - 大规模时可换 IndexIVFFlat 加速
"""

import os
import json
import re
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CORPUS_DIR   = "./law_corpus"
VECTOR_DIR   = "./vector_store"
# 使用ModelScope下载，避免HuggingFace网络问题
MODEL_NAME   = "BAAI/bge-small-zh-v1.5"
MODEL_CACHE  = "/root/autodl-tmp/embedding_model"


def load_and_chunk_corpus(corpus_dir: str) -> list[dict]:
    """
    加载法律语料并按条款切分。

    切分逻辑：
    以 《法律名称》第X条 为分隔符，
    每个条款 = {text: "《》第X条\n条文内容", source: "文件名", article: "第X条"}
    """
    chunks = []

    for filename in os.listdir(corpus_dir):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(corpus_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # 按《法律名》第X条分割
        # 正则：匹配 《任意内容》第中文数字条 作为分隔符
        pattern = r'(?=《[^》]+》第[零一二三四五六七八九十百千]+条)'
        parts = re.split(pattern, content)

        for part in parts:
            part = part.strip()
            if not part or len(part) < 20:  # 过滤空内容
                continue

            # 提取条款编号
            article_match = re.match(r'(《[^》]+》第[零一二三四五六七八九十百千]+条)', part)
            article_id = article_match.group(1) if article_match else "未知条款"

            chunks.append({
                "text":    part,
                "source":  filename,
                "article": article_id,
                "length":  len(part)
            })

    print(f"📄 共切分出 {len(chunks)} 个条款")
    for chunk in chunks[:3]:  # 展示前3个验证切分效果
        print(f"   [{chunk['article']}] {chunk['text'][:50]}...")

    return chunks


def download_embedding_model(model_name: str, cache_dir: str) -> SentenceTransformer:
    """
    下载并加载Embedding模型。
    优先从ModelScope下载（国内速度快）。
    """
    os.makedirs(cache_dir, exist_ok=True)

    print(f"⬇️  正在加载Embedding模型: {model_name}")
    print(f"   缓存目录: {cache_dir}")

    # 尝试从ModelScope下载
    try:
        from modelscope import snapshot_download
        local_path = snapshot_download(
            model_name,
            cache_dir=cache_dir
        )
        print(f"✅ 从ModelScope加载成功: {local_path}")
        model = SentenceTransformer(local_path)
    except Exception as e:
        print(f"⚠️  ModelScope失败({e})，尝试HuggingFace...")
        model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir
        )

    print(f"✅ 模型加载完成")
    print(f"   最大输入长度: {model.max_seq_length} tokens")
    return model


def build_faiss_index(chunks: list[dict], model: SentenceTransformer) -> faiss.Index:
    """
    将所有条款向量化并构建FAISS索引。
    """
    texts = [chunk["text"] for chunk in chunks]

    print(f"\n🔢 开始向量化 {len(texts)} 个条款...")
    start = time.time()

    # batch_size=32 适合CPU运行
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True  # L2归一化，使内积等价于余弦相似度
    )

    elapsed = time.time() - start
    print(f"✅ 向量化完成，耗时 {elapsed:.1f}s")
    print(f"   向量维度: {embeddings.shape[1]}")
    print(f"   平均速度: {len(texts)/elapsed:.1f} 条/秒")

    # 构建FAISS索引（IndexFlatIP = 内积搜索 = 余弦相似度搜索）
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    print(f"✅ FAISS索引构建完成，共 {index.ntotal} 条向量")
    return index


def save_index(index: faiss.Index, chunks: list[dict], vector_dir: str):
    """保存索引和元数据"""
    os.makedirs(vector_dir, exist_ok=True)

    # 保存FAISS索引
    index_path = os.path.join(vector_dir, "faiss.index")
    faiss.write_index(index, index_path)

    # 保存chunk元数据（用于检索后还原原文）
    meta_path = os.path.join(vector_dir, "chunks_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    index_size = os.path.getsize(index_path) / 1024
    print(f"\n💾 索引已保存:")
    print(f"   FAISS索引: {index_path} ({index_size:.1f} KB)")
    print(f"   元数据:    {meta_path}")


def verify_index(index: faiss.Index, chunks: list[dict], model: SentenceTransformer):
    """验证索引效果：用一个测试查询验证"""
    test_query = "公司裁员我能拿到多少赔偿"
    print(f"\n🔍 验证索引（测试查询：'{test_query}'）")

    query_vec = model.encode([test_query], normalize_embeddings=True)
    scores, indices = index.search(query_vec.astype(np.float32), k=3)

    print("Top-3 检索结果：")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        chunk = chunks[idx]
        print(f"  [{rank+1}] 相似度: {score:.4f} | {chunk['article']}")
        print(f"       {chunk['text'][:80]}...")


if __name__ == "__main__":
    print("=" * 50)
    print("开始构建法律RAG向量索引")
    print("=" * 50)

    # Step 1: 加载和切分语料
    chunks = load_and_chunk_corpus(CORPUS_DIR)

    # Step 2: 加载Embedding模型
    model = download_embedding_model(MODEL_NAME, MODEL_CACHE)

    # Step 3: 构建索引
    index = build_faiss_index(chunks, model)

    # Step 4: 保存
    save_index(index, chunks, VECTOR_DIR)

    # Step 5: 验证
    verify_index(index, chunks, model)

    print("\n✅ 索引构建完成！可以开始运行RAG Pipeline。")
