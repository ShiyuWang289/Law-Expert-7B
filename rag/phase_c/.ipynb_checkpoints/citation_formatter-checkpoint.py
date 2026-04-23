# phase_c/citation_formatter.py
"""
引用溯源格式化模块

工业级RAG必须支持引用溯源，原因：
1. 可信度：用户能核实回答的法律依据
2. 可调试：开发者能追溯"模型为何这么回答"
3. 对抗断章取义：展示原文片段，让误读可见

输出格式（结构化）：
{
  "answer": "根据《劳动合同法》...",
  "citations": [
    {
      "source_id":       "劳动合同法_第四十七条",
      "law_name":        "《劳动合同法》",
      "article":         "第四十七条",
      "snippet":         "经济补偿按劳动者在本单位工作的年限...",
      "relevance_score": 0.923,
      "rank":            1
    }
  ]
}

Prompt中的引用指令：
在System Prompt末尾加入强制引用要求，
让模型在回答末尾主动标注【来源：XXX】
"""

import re


def build_citation_prompt(
    base_system_prompt: str,
    retrieved_chunks:   list[dict]
) -> tuple[str, list[dict]]:
    """
    修复版：强制模型每个来源单独一行标注
    关键改动：引用规范里加了"每条来源必须单独占一行"的约束
    """
    if not retrieved_chunks:
        return base_system_prompt, []

    citation_metadata = []
    context_lines = ["【参考法条】（系统检索结果，请优先参考）\n"]

    for i, chunk in enumerate(retrieved_chunks):
        # ↓ 修复：清理 article 字段里可能重复的法律名称
        law_name_clean = chunk['source'].replace('.txt', '').replace('_', '')
        article_raw    = chunk.get("article", f"chunk_{i}")
        # 去掉 article 中如果包含了法律名的部分（如"劳动合同法第X条"→"第X条"）
        for law_prefix in ["劳动合同法", "劳动法", "工伤保险条例",
                           "民法典", "消费者权益保护法",
                           "劳动争议调解仲裁法", "劳动争议司法解释二"]:
            if article_raw.startswith(law_prefix):
                article_raw = article_raw[len(law_prefix):]
                break
        article_clean = re.sub(r'[《》\s]', '', article_raw)
        source_id = f"{law_name_clean}__{article_clean}"

        citation_metadata.append({
            "source_id":       source_id,
            "law_name":        chunk.get("law_name", law_name_clean),
            "article":         article_raw,
            "snippet":         chunk["text"][:150],
            "full_text":       chunk["text"],
            "relevance_score": chunk.get("rerank_score", chunk.get("rrf_score", 0.0)),
            "rank":            i + 1
        })

        context_lines.append(
            f"[来源{i+1}: {source_id}]\n{chunk['text']}\n"
        )

    context = "\n".join(context_lines)

    # ↓ 修复：引用规范更严格，每条来源必须单独一行
    citation_instruction = """

## 引用规范（必须严格遵守）
在回答末尾，每引用一条法律依据，必须单独占一行，格式如下：
【来源：劳动合同法__第四十七条】
【来源：劳动合同法__第四十八条】

禁止：【来源：A、B、C】（不允许多个来源写在同一个标签内）
如果引用了N条法律，就写N行【来源：...】标签。
未在参考法条中出现的依据，请注明：（模型知识，请核实）"""

    augmented_prompt = f"{base_system_prompt}{citation_instruction}\n\n{context}"
    return augmented_prompt, citation_metadata


def extract_citations(
    model_answer:      str,
    citation_metadata: list[dict]
) -> dict:
    """
    修复版：支持每行一个来源的格式
    同时兼容模型偶尔用逗号/顿号分隔的情况（降级处理）
    """
    pattern = r'【来源[：:]\s*([^\】]+)】'
    raw_matches = re.findall(pattern, model_answer)

    # 展开：如果模型还是用了逗号/顿号，拆开处理（降级兼容）
    cited_ids = []
    for match in raw_matches:
        # 按中文顿号、逗号、英文逗号分割
        parts = re.split(r'[、，,]', match)
        cited_ids.extend([p.strip() for p in parts if p.strip()])

    matched_citations = []
    unmatched_ids     = []

    for cited_id in cited_ids:
        match = next(
            (m for m in citation_metadata if m["source_id"] == cited_id),
            None
        )
        if match:
            matched_citations.append(match)
        else:
            # 尝试模糊匹配（去掉空格后对比）
            fuzzy_match = next(
                (m for m in citation_metadata
                 if m["source_id"].replace(" ", "") == cited_id.replace(" ", "")),
                None
            )
            if fuzzy_match:
                matched_citations.append(fuzzy_match)
            else:
                unmatched_ids.append(cited_id)

    return {
        "matched":       matched_citations,
        "unmatched":     unmatched_ids,
        "total_cited":   len(cited_ids),
        "citation_rate": len(matched_citations) / max(len(citation_metadata), 1)
    }


def format_final_output(
    question:          str,
    answer:            str,
    citation_metadata: list[dict],
    latency:           float
) -> dict:
    """
    构建最终结构化输出（面试展示用）
    """
    citation_result = extract_citations(answer, citation_metadata)

    return {
        "question": question,
        "answer":   answer,
        "citations": [
            {
                "source_id":       c["source_id"],
                "article":         c["article"],
                "snippet":         c["snippet"],
                "relevance_score": round(c["relevance_score"], 4),
                "rank":            c["rank"]
            }
            for c in citation_result["matched"]
        ],
        "citation_stats": {
            "retrieved_count":  len(citation_metadata),
            "cited_count":      citation_result["total_cited"],
            "matched_count":    len(citation_result["matched"]),
            "citation_rate":    round(citation_result["citation_rate"], 3),
            "unmatched_ids":    citation_result["unmatched"]
        },
        "latency_s": round(latency, 2)
    }
    