# phase_a/chunking_strategies.py
"""
三种切分策略对比实验

策略A：按条款边界切分（当前方式，作为基线）
策略B：固定字数切分（200字+30字重叠）
策略C：RecursiveCharacterTextSplitter（语义优先）

设计思路：
切分策略决定了"检索的最小单元"
单元太大 → 一个chunk混合多个主题 → 检索不精准（TC003的根本原因）
单元太小 → 上下文不完整 → LLM缺乏足够信息生成回答
"""

import os
import re
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

CORPUS_DIR = "../law_corpus"
OUTPUT_DIR = "./chunk_results"


# ============================================================
# 策略A：按条款边界切分（基线，与现有方式相同）
# ============================================================
def strategy_a_by_article(corpus_dir: str) -> list[dict]:
    """
    以《法律名》第X条为边界进行切分。
    每个条款 = 一个chunk。
    这是当前retriever.py中02_build_index.py使用的方式。
    """
    chunks = []
    pattern = r'(?=《[^》]+》第[零一二三四五六七八九十百千]+条)'

    for filename in sorted(os.listdir(corpus_dir)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(corpus_dir, filename), encoding="utf-8") as f:
            content = f.read()

        parts = re.split(pattern, content)
        for part in parts:
            part = part.strip()
            if len(part) < 20:
                continue
            article_match = re.match(r'(《[^》]+》第[零一二三四五六七八九十百千]+条)', part)
            chunks.append({
                "text":     part,
                "source":   filename,
                "article":  article_match.group(1) if article_match else "未知",
                "strategy": "A_by_article",
                "length":   len(part)
            })

    return chunks


# ============================================================
# 策略B：固定字数切分（200字 + 30字重叠）
# ============================================================
def strategy_b_fixed_size(corpus_dir: str) -> list[dict]:
    """
    固定200字切分，30字重叠。

    设计理由：
    - 200字约等于150 token，远低于bge-small的512上限
    - 30字重叠防止关键信息被切断在边界
    - 每个chunk语义更集中

    已知问题：
    - 可能把一个条款切成多个chunk（如第四十一条共400字会被切成2份）
    - 跨chunk的条款之间没有关联

    chunk_id格式：filename_序号，用于溯源
    """
    chunks = []

    for filename in sorted(os.listdir(corpus_dir)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(corpus_dir, filename), encoding="utf-8") as f:
            content = f.read()

        chunk_size    = 200
        chunk_overlap = 30
        step          = chunk_size - chunk_overlap

        for i, start in enumerate(range(0, len(content), step)):
            text = content[start: start + chunk_size].strip()
            if len(text) < 30:
                continue
            chunks.append({
                "text":     text,
                "source":   filename,
                "article":  f"{filename}_chunk_{i}",
                "strategy": "B_fixed_size",
                "length":   len(text)
            })

    return chunks


# ============================================================
# 策略C：RecursiveCharacterTextSplitter（语义优先）
# ============================================================
def strategy_c_recursive(corpus_dir: str) -> list[dict]:
    """
    使用LangChain的RecursiveCharacterTextSplitter。

    分隔符优先级（从高到低尝试）：
    1. 条款边界（《法律》第X条）→ 优先按条款切
    2. 双换行（段落边界）
    3. 单换行
    4. 句号（句子边界）
    5. 字符级（兜底）

    最大chunk_size=300字（平衡长度和完整性）
    overlap=50字（比固定切分更大，因为递归切分的边界更随机）

    这是工业界法律RAG最常用的切分方式：
    - 优先保留语义完整单元（条款）
    - 当条款太长时退化到段落级别切分
    - 永远不会产生太短的无意义片段
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=[
            r'(?=《[^》]+》第[零一二三四五六七八九十百千]+条)',  # 条款边界
            "\n\n",   # 段落
            "\n",     # 换行
            "。",     # 句子
            "",       # 字符级兜底
        ],
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=True,   # 第一个分隔符是正则
    )

    chunks = []
    for filename in sorted(os.listdir(corpus_dir)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(corpus_dir, filename), encoding="utf-8") as f:
            content = f.read()

        docs = splitter.create_documents(
            [content],
            metadatas=[{"source": filename}]
        )

        for i, doc in enumerate(docs):
            text = doc.page_content.strip()
            if len(text) < 20:
                continue
            article_match = re.search(
                r'(《[^》]+》第[零一二三四五六七八九十百千]+条)', text
            )
            chunks.append({
                "text":     text,
                "source":   filename,
                "article":  article_match.group(1) if article_match else f"{filename}_seg_{i}",
                "strategy": "C_recursive",
                "length":   len(text)
            })

    return chunks


# ============================================================
# 统计分析：对比三种策略的基本指标
# ============================================================
def analyze_chunks(chunks: list[dict], strategy_name: str) -> dict:
    lengths = [c["length"] for c in chunks]
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    min_len = min(lengths)

    # 统计"过长chunk"（>400字，超过bge推荐输入长度）
    too_long = sum(1 for l in lengths if l > 400)
    # 统计"过短chunk"（<50字，信息量不足）
    too_short = sum(1 for l in lengths if l < 50)

    stats = {
        "strategy":   strategy_name,
        "total":      len(chunks),
        "avg_length": round(avg_len, 1),
        "max_length": max_len,
        "min_length": min_len,
        "too_long":   too_long,    # >400字的chunk数
        "too_short":  too_short,   # <50字的chunk数
    }

    print(f"\n策略{strategy_name}统计：")
    print(f"  chunk总数: {stats['total']}")
    print(f"  平均长度: {stats['avg_length']} 字")
    print(f"  最长: {stats['max_length']} 字 | 最短: {stats['min_length']} 字")
    print(f"  过长chunk(>400字): {too_long} 个 | 过短chunk(<50字): {too_short} 个")

    return stats


def save_chunks(chunks: list[dict], filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("文档切分策略对比实验")
    print("=" * 60)

    all_stats = []

    for strategy_fn, name, save_name in [
        (strategy_a_by_article, "A（按条款）", "chunks_A.json"),
        (strategy_b_fixed_size, "B（固定200字）", "chunks_B.json"),
        (strategy_c_recursive,  "C（递归语义）", "chunks_C.json"),
    ]:
        chunks = strategy_fn(CORPUS_DIR)
        stats  = analyze_chunks(chunks, name)
        save_chunks(chunks, save_name)
        all_stats.append(stats)

    # 打印汇总对比表
    print("\n" + "=" * 60)
    print("汇总对比：")
    print(f"{'策略':<20} {'总数':>6} {'均长':>6} {'最长':>6} {'过长':>6} {'过短':>6}")
    print("-" * 60)
    for s in all_stats:
        print(f"{s['strategy']:<20} {s['total']:>6} {s['avg_length']:>6} "
              f"{s['max_length']:>6} {s['too_long']:>6} {s['too_short']:>6}")