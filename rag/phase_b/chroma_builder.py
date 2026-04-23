# phase_b/chroma_builder.py
"""
用Chroma替代FAISS构建向量数据库。

Chroma相比FAISS的核心优势（对本项目有用的）：
1. 支持metadata过滤：按law_type筛选，减少TC009类噪声
2. 持久化简单：自动管理数据文件
3. 原生支持混合检索（后续Phase C用）

metadata设计：
每个chunk存储：
  - source: 文件名（原有）
  - article: 条款编号（原有）
  - law_type: 法律类型（新增，用于过滤）
    劳动合同法/工伤保险条例 → "劳动"
    民法典_侵权 → "侵权"
    民法典_合同 → "合同"
    消费者权益保护法 → "消费"
  - law_name: 法律简称（新增）
"""

import os
import sys
import json
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

sys.path.append("../phase_b")

MODEL_CACHE  = "/root/autodl-tmp/embedding_model"
CHROMA_DIR   = "../vector_store/chroma_db"

# 法律类型映射（文件名 → law_type标签）
LAW_TYPE_MAP = {
    "劳动合同法.txt":       "劳动",
    "劳动法.txt":           "劳动",
    "工伤保险条例.txt":     "工伤",
    "劳动争议司法解释二.txt": "劳动",
    "劳动争议调解仲裁法.txt": "劳动",
    "民法典_侵权.txt":      "侵权",
    "民法典_合同.txt":      "合同",
    "消费者权益保护法.txt": "消费",
    "典型案例_新业态用工.txt": "劳动",
    "经济补偿实务问答.txt": "劳动",
}


def find_best_model() -> str:
    """找到本地最好的Embedding模型（优先bge-base）"""
    for keyword in ["bge-base-zh", "bge-small-zh"]:
        for root, dirs, files in os.walk(MODEL_CACHE):
            if "config.json" in files and keyword in root.lower():
                return root
    raise FileNotFoundError("找不到Embedding模型")


def build_chroma_index(chunks: list[dict]) -> chromadb.Collection:
    """
    构建Chroma向量数据库。
    使用sentence-transformers作为embedding函数。
    """
    model_path = find_best_model()
    print(f"使用模型: {model_path}")

    # Chroma的embedding函数封装
    model = SentenceTransformer(model_path)

    class CustomEmbeddingFunction(embedding_functions.EmbeddingFunction):
        def __call__(self, texts):
            return model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False
            ).tolist()

    # 初始化Chroma客户端（持久化模式）
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # 删除已有collection（重新构建）
    try:
        client.delete_collection("law_chunks")
        print("已删除旧collection")
    except:
        pass

    collection = client.create_collection(
        name="law_chunks",
        embedding_function=CustomEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"}  # 余弦相似度
    )

    # 批量插入（Chroma建议每批最多500条）
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]

        ids        = [f"chunk_{i+j}" for j in range(len(batch))]
        documents  = [c["text"] for c in batch]
        metadatas  = [
            {
                "source":   c["source"],
                "article":  c["article"],
                "law_type": LAW_TYPE_MAP.get(c["source"], "其他"),
                "law_name": c["source"].replace(".txt", ""),
                "length":   c["length"],
            }
            for c in batch
        ]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"  已插入 {min(i+batch_size, len(chunks))}/{len(chunks)} 条")

    print(f"\n✅ Chroma索引构建完成")
    print(f"   存储路径: {CHROMA_DIR}")
    print(f"   总条数: {collection.count()}")

    return collection


def test_metadata_filter(collection: chromadb.Collection, model_path: str):
    """
    测试metadata过滤功能。
    核心目的：验证TC009问题（名誉权）在过滤后不再误检劳动合同法
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_path)

    test_cases = [
        {
            "question":    "同事在公司群里发消息说我贪污，完全是造谣，我能告他吗？",
            "filter":      {"law_type": {"$in": ["侵权", "合同"]}},  # 过滤掉劳动法
            "description": "TC009名誉权（加过滤）",
        },
        {
            "question":    "同事在公司群里发消息说我贪污，完全是造谣，我能告他吗？",
            "filter":      None,  # 不过滤，对比用
            "description": "TC009名誉权（无过滤）",
        },
    ]

    query_vec = model.encode(
        [tc["question"] for tc in test_cases],
        normalize_embeddings=True
    )

    for i, tc in enumerate(test_cases):
        print(f"\n{tc['description']}：")
        kwargs = {
            "query_embeddings": [query_vec[i].tolist()],
            "n_results": 3,
            "include": ["metadatas", "distances", "documents"]
        }
        if tc["filter"]:
            kwargs["where"] = tc["filter"]

        results = collection.query(**kwargs)
        for j, (meta, dist) in enumerate(
            zip(results["metadatas"][0], results["distances"][0])
        ):
            print(f"  [{j+1}] {meta['law_name']} | {meta['article']} | 距离:{dist:.4f}")


if __name__ == "__main__":
    # 使用Phase A最优切分策略的chunks
    with open("../phase_a/chunk_results/chunks_C.json") as f:
        chunks = json.load(f)

    collection = build_chroma_index(chunks)
    model_path = find_best_model()
    test_metadata_filter(collection, model_path)