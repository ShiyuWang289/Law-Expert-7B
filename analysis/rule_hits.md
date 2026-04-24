# 清洗规则命中统计（P2）

- 输入文件: `data/law_qa_alpaca.json`
- 输出文件: `raw_data/law_qa_cleaned.jsonl`

## 阶段统计

| 阶段 | 样本数 |
|---|---:|
| 原始样本 | 483 |
| 基础清洗后 | 483 |
| 质量过滤输入 | 483 |
| 质量过滤后 | 481 |
| 质量阶段删除量 | 2 |

## 质量规则命中明细

| 规则Key | 规则 | 命中数 | 在质量阶段命中比例 | 对原始总量删除贡献 |
|---|---|---:|---:|---:|
| low_legal_relevance | no legal keywords in output | 2 | 0.41% | 0.41% |
