# Law-Expert-7B: 基于Qwen2.5的全链路微调法律领域专家模型

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本项目完整复现了将一个通用大语言模型（LLM）通过全链路微调技术（SFT + DPO）适配到法律垂直领域，并最终实现高性能服务化部署的全过程。旨在探索和验证一套低成本、高效率的LLM领域化最佳实践。

## 🚀 项目亮点

- **全链路实践:** 覆盖从数据处理、SFT指令微调、DPO偏好对齐，到模型评测、量化、vLLM部署的工业级全流程。
- **数据驱动:** 采用精细化数据清洗策略，构建高质量指令与偏好数据集，验证“数据质量远胜于数量”的核心原则。
- **性能卓越:** 微调后模型在法律领域关键指标上提升显著，同时通用能力无“灾难性遗忘”，并通过vLLM实现**2.4倍**的推理加速。
- **高复用性:** 整体方案基于`LlamaFactory`和`vLLM`等主流框架，具备高度可复现性和可迁移性。

---

## 📜 项目背景

法律领域对信息的准确性、专业性和时效性有极高要求。通用大模型虽知识广博，但在专业领域的深度和表达风格上常有不足（如引用法条不准、表述过于宽泛）。本项目旨在通过高效参数微调（PEFT）技术，以较低的算力成本，将`Qwen2.5-7B-Instruct`模型改造为法律问答专家`Law-Expert-7B`，使其回答更精准、专业，更贴近真实法律咨询场景。

## 📊 数据处理

高质量的数据是模型成功的基石。我们遵循“数据质量 >> 数据数量”的原则，构建了指令微调和偏好微调所需的数据集。

- **数据来源:** 原始数据来源于 [DISC-Law-SFT](https://huggingface.co/datasets/DISC-UI/DISC-Law-SFT)，包含约9万条法律问答数据。

- **清洗与构建策略:**
  1.  **SFT指令数据 (Alpaca格式):**
      - **四层清洗:** 实施了包括基础清洗（去空、去重）、质量过滤（长度限制、移除噪声）、多样性优化（前缀去重）和一致性检查的精细化清洗流程。
      - **精选采样:** 从9万条原始数据中，最终精选出 **约500条** 高质量单轮问答样本，统一为Alpaca格式。
  2.  **DPO偏好数据 (Chosen/Rejected):**
      - **策略:** 结合**手工编写**和**自动构建**两种方式。
      - **手工编写 (8条):** 针对核心法律场景，手工编写高质量的`chosen`（法条准确、逻辑完整）和`rejected`（法条错误、回答模糊）对。
      - **自动构建 (50条):** 从SFT数据中，将原始高质量`output`作为`chosen`，通过截断、模糊化等方式生成对应的`rejected`。
      - **最终数量:** 共计 **约58条** 偏好数据对。

## ⚙️ 训练配置

我们采用了SFT -> DPO的二级微调策略。

- **基座模型:** `Qwen/Qwen2.5-7B-Instruct`
  - **选择理由:** 中文能力强，社区活跃，作为Instruct版本已具备良好对话基础，适合在其上进行领域增强。

- **SFT阶段 (QLoRA):**
  - **目标:** 让模型学习法律领域的知识和基础问答风格。
  - **核心参数:**
| 参数 | 值 | 说明 |
| :--- | :--- | :--- |
| `quantization_bit` | 4 | 使用4-bit量化，即QLoRA |
| `lora_rank` | 16 | LoRA矩阵的秩，平衡性能与参数量 |
| `lora_alpha` | 32 | LoRA缩放因子，通常设为rank的2倍 |
| `lora_target` | `all` | 对模型所有线性层应用LoRA |
| `learning_rate` | `1e-4` | SFT阶段的学习率 |
| `per_device_train_batch_size` | 2 | - |
| `gradient_accumulation_steps`| 8 | 等效batch size为 16 |
| `cutoff_len` | 2048 | 样本最大长度，覆盖95%以上数据 |

- **DPO阶段:**
  - **目标:** 让模型学习人类偏好，使其回答更符合“好的法律咨询”的标准（更详尽、更准确）。
  - **核心参数:**
| 参数 | 值 | 说明 |
| :--- | :--- | :--- |
| `ref_model` | SFT阶段产出的Adapter | 在SFT模型基础上继续训练 |
| `pref_beta` | 0.1 | DPO的正则化强度，控制与`ref_model`的偏离程度 |
| `learning_rate` | `5e-6` | DPO阶段学习率，通常比SFT小一个数量级 |
| `create_new_adapter` | `true` | 为DPO训练创建新的Adapter，不覆盖SFT的 |

## 📈 评测结果

我们从**通用能力**和**领域能力**两个维度对模型进行量化评估。

| 模型 | C-Eval (通用能力) | 法律关键点覆盖率 (领域能力) |
| :--- | :---: | :---: |
| **Base** (Qwen2.5-7B-Instruct) | 78.83% | *较低* |
| **SFT** (本项目) | **79.42%** (+0.59%) | 13.3% |
| **DPO** (本项目) | 79.05% (+0.22%) | **16.7%** (+3.4% vs SFT) |

**结论分析:**
1.  **领域能力显著提升:** 经过SFT+DPO微调后，模型在法律案例关键法条的覆盖率上从基线的13.3%提升至**16.7%**，DPO阶段带来了关键的质量飞跃。
2.  **通用能力无损:** 在权威的中文通用能力评测集C-Eval上，SFT和DPO模型得分均略有提升，证明本次微调**未导致灾难性遗忘**，模型通用知识得以保留。
3.  **定性效果:** 对比发现，DPO模型在回答时，法条引用更**准确具体**，回答逻辑更**完整**，并倾向于给出**完整的维权路径**（协商->仲裁->诉讼），整体风格更接近专业法律咨询。

## 🖥️ 部署方案

为了在生产环境中实现高效推理，我们设计了以下部署方案：

1.  **模型合并与量化:**
    - 使用`llamafactory-cli export`命令将SFT和DPO的LoRA权重依次合并到基座模型中。
    - 使用`llm-compressor`工具对合并后的模型进行`W4A16`（Weight-4bit, Activation-16bit）量化，模型体积从 **~15GB** 压缩至 **~5.2GB**。

2.  **推理服务:**
    - **框架:** `vLLM`
    - **核心技术:** 利用其`PagedAttention`和`Continuous Batching`特性，大幅提升GPU利用率和吞吐量。
    - **启动命令:**
      ```bash
      vllm serve ./path/to/merged_quantized_model \
          --served-model-name "law-expert" \
          --host 0.0.0.0 \
          --port 6006 \
          --dtype auto \
          --max-model-len 4096 \
          --gpu-memory-utilization 0.9 \
          --tensor-parallel-size 1
      ```

3.  **API设计:**
    - `vLLM`提供与OpenAI完全兼容的API接口，现有应用可**零代码**迁移。
    - **Python调用示例:**
      ```python
      from openai import OpenAI

      # 只需修改 base_url 和 api_key
      client = OpenAI(
          api_key="EMPTY",
          base_url="http://localhost:6006/v1"
      )

      response = client.chat.completions.create(
          model="law-expert",
          messages=[{"role": "user", "content": "劳动合同到期公司不续签，有赔偿吗？"}],
          stream=True
      )

      for chunk in response:
          print(chunk.choices[0].delta.content or "", end="")
      ```

## ⚡ 性能指标

在`Tesla V100S-32GB`环境下实测性能数据如下：

| 指标 | 数值 | 说明 |
| :--- | :--- | :--- |
| 推理引擎 | vLLM | - |
| 模型精度 | FP16 (W4A16加载) | W4A16在推理时反量化为FP16进行计算 |
| **推理速度** | **48.5 tokens/s** | - |
| Transformers基线 | ~20.0 tokens/s | 对比`transformers`原生`pipeline` |
| **加速比** | **2.4x** | vLLM带来的显著性能提升 |
| 平均响应时间 | 2.64s / 请求 | 平均每个回答（128 tokens）的耗时 |

---
# 🎯 Prompt Engineering 实验

在模型微调完成后，我们进行了一系列Prompt Engineering实验，旨在探索在已充分微调的模型上，不同Prompt策略对输出质量的优化效果。这代表了LLM应用中成本最低、迭代最快的“最后一公里”优化。

### 实验设计

- **固定变量**: `law-expert-dpo`模型、`temperature=0.1`。
- **测试集**: 10个覆盖劳动法、合同法、侵权等领域的典型案例。
- **可变因素**: 4个版本的System Prompt。

| 版本 | 核心策略 | 设计目的 |
|:---|:---|:---|
| **v1_baseline** | `你是一个法律专家...` | 验证模型微调后的“默认行为” |
| **v2_structured** | 结构化指令（角色、规范、禁止项） | 检验强约束对输出格式的控制能力 |
| **v3_cot_lite** | 轻量化思维链 + 弹性指令 | 探索在提升复杂问题推理能力的同时，避免简单问题“过度思考” |
| **v4_fewshot** | One-shot示例 | 通过高质量范例引导模型的输出风格和内容深度 |

### 实验结果

我们设计了包含关键点覆盖率、法条格式规范性、回答结构完整性等多维度的自动化评测脚本。

| Prompt 版本 | 平均总分 | 关键点覆盖率 | 法条引用数 | 平均响应时间 |
|:---|:---:|:---:|:---:|:---:|
| v1_baseline | 84.5 / 100 | 60.7% | 7.3 条 | 9.9s |
| **v2_structured** | **89.8 / 100** | 59.3% | **11.1 条** | 12.5s |
| v3_cot_lite | 79.0 / 100 | 58.3% | 3.6 条 | **5.5s** |
| v4_fewshot | 83.9 / 100 | 54.3% | 2.7 条 | **5.0s** |

*（注：此为v2实验数据，分数存在随机波动，但趋势具有代表性）*

### 关键发现与结论

1.  **结构化约束（v2）是稳健最优解**: 清晰的角色定义、行为规范和格式要求，在不显著增加延迟的情况下，全面提升了模型的专业性和可靠性，特别是在“法条格式”维度得分最高。

2.  **CoT（v3）对小模型是双刃剑**: 轻量化CoT虽修复了原版在简单问题上的退步，但因过度精简指令导致在其他问题上表现不佳。这证明CoT策略需要根据问题复杂度动态路由，不适合“一刀切”全局应用。

3.  **Few-shot（v4）在已微调模型上收益有限**: 对于已通过SFT掌握领域风格的模型，Few-shot的引导作用被SFT的“肌肉记忆”所覆盖，其边际收益低于结构化指令，且存在示例内容干扰的风险。

4.  **Prompt优化的是“形式”而非“知识”**: 实验表明，Prompt能显著改变输出的格式和结构（怎么说），但对内容的准确性（知道什么）提升有限。模型的知识边界主要由Finetune阶段决定。

### 工程实践建议

- **默认选项**: **v2_structured** prompt是兼顾质量、稳定性和成本的最佳基线。
- **场景化路由**: 针对需要深度逻辑推理的复杂场景，可设计路由策略，动态切换到CoT类Prompt。
- **迭代顺序**: LLM应用优化应遵循“**Finetune → Prompt → RAG**”的顺序。在Finetune奠定知识基础后，用Prompt做格式和风格的精调，最后用RAG解决知识时效性问题。

  ```markdown
# 🎯 法律 RAG 实践

此部分基于 LLaMA-Factory 微调的法律模型 `law-expert`，构建了一套完整的 RAG（检索增强生成）系统，用于法律咨询场景。项目覆盖从文档切分、Embedding 选型、向量数据库构建，到 Hybrid 检索、Reranker 精排、引用溯源的完整链路，并通过消融实验量化各组件贡献。

## 目录结构

```
rag/
├── law_corpus/                 # 原始法律文本语料（劳动法、民法典等）
├── vector_store/               # 向量数据库
│   ├── faiss.index             # FAISS 索引（V2 使用）
│   ├── chroma_db/              # Chroma 持久化目录（含元数据过滤）
│   └── bm25_index.pkl          # BM25 稀疏索引
├── phase_a/                    # 切分策略对比
│   ├── chunking_strategies.py
│   ├── evaluate_chunking.py
│   └── results/
├── phase_b/                    # Embedding 选型 + Chroma 构建
│   ├── embedding_benchmark.py
│   ├── chroma_builder.py
│   └── results/
├── phase_c/                    # 企业级组件（Hybrid/Reranker/引用）
│   ├── bm25_index.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   ├── citation_formatter.py
│   └── enterprise_pipeline.py
├── phase_d/                    # 消融实验
│   ├── ablation_experiment.py
│   └── results/
├── retriever.py                # 基础检索器（V2）
├── rag_pipeline.py             # 基础 RAG 问答
└── requirements_rag_v2.txt     # 依赖清单
```

## 快速开始

### 1. 环境准备

```bash
cd /root/autodl-tmp/LLaMA-Factory
source .venv/bin/activate

# 安装依赖
pip install -r rag/requirements_rag_v2.txt
```

### 2. 数据准备与索引构建

```bash
# Phase A：生成三种切分策略并评测（默认选用策略 C）
cd rag/phase_a
python chunking_strategies.py
python evaluate_chunking.py

# Phase B：构建 Chroma 向量库（含元数据过滤）
cd ../phase_b
python chroma_builder.py

# Phase C：构建 BM25 索引
cd ../phase_c
python bm25_index.py
```

### 3. 启动 vLLM 推理服务

```bash
vllm serve /root/autodl-tmp/merged_models/law_qa_dpo \
    --served-model-name law-expert \
    --port 6006 \
    --dtype float16 \
    --max-model-len 2048
```

### 4. 运行完整 Pipeline

```bash
cd rag/phase_c
python enterprise_pipeline.py   # 测试 RAG 问答
```

### 5. 消融实验（量化组件贡献）

```bash
cd ../phase_d
python ablation_experiment.py   # 生成 V1~V5 对比报告
cat results/ablation_report.md
```

## 核心技术栈

| 层级 | 组件 | 说明 |
|:---|:---|:---|
| 切分 | RecursiveCharacterTextSplitter | 优先按条款边界切分，chunk_size=300，overlap=50 |
| Embedding | bge-base-zh-v1.5 | 768 维，区分度 gap 达 0.0551 |
| 向量库 | Chroma + FAISS | Chroma 支持元数据过滤（按 law_type），FAISS 作为简单基线 |
| 稀疏检索 | BM25 (rank-bm25) | jieba 分词，提供精确词汇匹配信号 |
| 融合 | RRF (k=60) | 无需调参的排名融合算法 |
| 精排 | bge-reranker-base | Cross-Encoder，CPU 推理，从 top-20 精选 top-3 |
| 引用溯源 | Prompt 强制标注 | 格式：【来源：劳动合同法__第四十七条】，后处理解析 |
| 推理引擎 | vLLM | 部署微调后的 Qwen2.5-7B 法律模型 |

## 消融实验结论（16 个测试案例）

| 版本 | 配置 | 平均总分 | 关键点覆盖 | 延迟 |
|:---|:---|:---:|:---:|:---:|
| V1 | 纯微调，无 RAG | 89.4 | 65.3% | 10.6s |
| V2 | 简单 RAG（bge-small + FAISS） | 90.3 | 64.3% | 7.8s |
| **V3** | **优化切分 + bge-base（纯 Dense）** | **91.7** | **68.8%** | **8.7s** |
| V4 | V3 + BM25 + RRF + Reranker | 86.4 | 58.8% | 14.9s |
| V5 | V4 + 引用溯源 | 85.8 | 58.3% | 14.1s |

> **结论**：在当前法律咨询语料上，**纯 Dense 检索（V3）综合表现最优**。BM25 在描述性查询中引入噪声（如 TC009 从 100 分降至 74.3 分），Reranker 带来的精度提升未能抵消其延迟成本。建议根据查询类型动态选择检索策略。

## 关键文件说明

- `phase_c/enterprise_pipeline.py`：完整问答入口，整合 Hybrid、Reranker、引用。
- `phase_c/hybrid_retriever.py`：支持 Chroma 元数据过滤的 Hybrid 检索器。
- `phase_c/reranker.py`：封装 bge-reranker-base，支持离线模式加载本地模型。
- `phase_d/ablation_experiment.py`：五版本对比脚本，直接生成 Markdown 报告。

## 常见问题

**Q：BM25 为什么在实验中导致分数下降？**  
A：用户查询多为日常描述（如“造谣贪污”），法律条文中不存在这些词汇，BM25 返回的文档与问题无关，污染了 RRF 融合结果。

**Q：如何启用 Chroma 元数据过滤？**  
A：在 `HybridRetriever.retrieve()` 中传入 `law_type_filter=["侵权", "合同"]` 即可，TC009 案例已验证其有效性。

**Q：Reranker 为什么放在 CPU 上？**  
A：V100S 显存需留给 vLLM，且 Reranker 仅处理 20 条候选，CPU 推理延迟可接受。

**Q：引用匹配率仅 13.3% 如何提升？**  
A：加强 Prompt 约束（如“每条来源必须单独占一行”），并在后处理中添加兜底自动补全逻辑。


```
  
