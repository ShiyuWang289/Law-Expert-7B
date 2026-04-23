# RAG系统消融实验报告

## 总体对比

| 版本 | 说明 | 平均总分 | 关键点覆盖 | 延迟 |
|:---|:---|:---:|:---:|:---:|
| V1_no_rag | 纯微调，无RAG（基线） | 89.4 | 65.3% | 10.6s |
| V2_simple_rag | 简单RAG（bge-small+FAISS） | 90.3 | 64.3% | 7.8s |
| V3_dense_bgebase | 优化切分+bge-base（纯Dense） | 91.7 | 68.8% | 8.7s |
| V4_hybrid_reranker | V3+BM25+RRF+Reranker | 86.4 | 58.8% | 14.9s |
| V5_full_citation | V4+引用溯源（完整版） | 85.8 | 58.3% | 14.1s |

## 组件贡献分析

| 对比 | 新增组件 | 分数变化 | 结论 |
|:---|:---|:---:|:---|
| V1_no_rag→V2_simple_rag | RAG检索 | +0.9 ↑ | 有效 |
| V2_simple_rag→V3_dense_bgebase | bge-base+优化切分 | +1.4 ↑ | 有效 |
| V3_dense_bgebase→V4_hybrid_reranker | BM25+RRF+Reranker | -5.3 ↓ | 有害 |
| V4_hybrid_reranker→V5_full_citation | 引用溯源 | -0.5 ↓ | 有害 |

## 引用溯源统计（V5）

- 平均引用匹配率：**20.0%**
- 目标：>60%