# phase_d/diagnose_retrieval.py
"""
诊断V2/V3/V4/V5的检索质量
"""

import sys
import json
from pathlib import Path

PHASE_D_DIR = Path(__file__).parent.absolute()
RAG_DIR = PHASE_D_DIR.parent
LLAMA_FACTORY_ROOT = RAG_DIR.parent

sys.path.insert(0, str(LLAMA_FACTORY_ROOT))
sys.path.insert(0, str(RAG_DIR))

from prompt_engineering.test_cases import TEST_CASES
from phase_c.enterprise_pipeline import EnterpriseLawRAG
from retriever import LawRetriever


def diagnose():
    """诊断检索质量"""
    print("🔍 ���索质量诊断\n")

    # 加载chunks
    chunks_path = RAG_DIR / "phase_a" / "chunk_results" / "chunks_C.json"
    with open(chunks_path, encoding="utf-8") as f:
        chunks_v3 = json.load(f)

    # 初始化检索器
    vector_store_path = RAG_DIR / "vector_store"
    retriever_v2 = LawRetriever(vector_dir=str(vector_store_path))
    enterprise_rag = EnterpriseLawRAG(chunks_v3, offline_mode=True)

    # 测试几个关键问题
    test_questions = [
        TEST_CASES[0],  # TC001
        TEST_CASES[2],  # TC003
        TEST_CASES[8],  # TC009 (问题问题)
    ]

    for tc in test_questions:
        print(f"\n{'='*60}")
        print(f"问题: {tc['id']} - {tc['question']}")
        print(f"{'='*60}")

        # V2检索
        print("\n【V2 - bge-small + FAISS】")
        v2_results = retriever_v2.retrieve(tc["question"], top_k=3)
        for i, r in enumerate(v2_results, 1):
            print(f"  {i}. [{r['score']:.4f}] {r['source']}")
            print(f"     {r['text'][:60]}...")

        # V3/V4/V5检索
        print("\n【V3/V4/V5 - Hybrid检索】")
        v3_candidates = enterprise_rag.hybrid.retrieve(tc["question"], top_k=20, candidate_k=20)
        print(f"  检索到{len(v3_candidates)}个候选")
        for i, c in enumerate(v3_candidates[:3], 1):
            print(f"  {i}. {c['source']}")
            print(f"     {c['text'][:60]}...")

        # Reranker精排
        print("\n【V4/V5 - Reranker精排】")
        reranked = enterprise_rag.reranker.rerank(tc["question"], v3_candidates, top_k=3)
        for i, r in enumerate(reranked, 1):
            print(f"  {i}. [{r['rerank_score']:.4f}] {r['source']}")


if __name__ == "__main__":
    diagnose()