---

# Law-Expert-7B：从 SFT 到 DPO、评测、部署的法律专家模型工程实践

---

## 0. 项目概览

### 项目目标

基于 Qwen2.5-7B-Instruct，通过 QLoRA + DPO 完成法律问答领域化改造，并将其做成**可持续迭代的工程闭环**：

- **数据工程**：可复现对照实验证明采样策略有效性
- **训练对齐**：SFT 解决输出分布迁移，DPO 两阶段完成偏好对齐（PoC 70对 → 正式约2,000对）
- **回归评测**：Golden 门禁 pipeline 保证每次迭代可告警、可定位
- **服务化部署**：vLLM FP16 部署 + 两段式压测，固化生产配置
- **扩展验证**：RAG 检索质量边界分析 + 轻量 Agent 闭环可观测性验证

### 技术路线总览

```
原始数据（DISC-Law-SFT ~9万）
    ↓ P1/P2 基础清洗
候选池 pool（67,261条）
    ↓ P3 对照实验（A随机500 vs B工程化500）
PoC SFT（B组500条，r=16 快速验证）→ PoC DPO（70对，pipeline验证）
    ↓ 策略确认后扩规模
正式SFT（~2万条，r=64） → 正式DPO（~2,000对，hard-negative）
    ↓
Golden 回归评测（30题门禁集，五指标差分告警）
    ↓
vLLM FP16 服务化 → 两段式压测（阶梯并发 + Locust稳定性）
    ↓ 扩展验证
RAG 检索质量边界 + 轻量 Agent 可观测闭环
```

### 当前进度看板

| 模块 | 状态 |
|------|------|
| P1 数据统计看板 | ✅ 已完成 |
| P2 清洗规则工程化（四元组） | ✅ 已完成 |
| P3 对照实验（随机500 vs 工程化500） | ✅ 已完成 |
| P4 学习率小型对照（lr ablation） | ✅ 已完成 |
| P5 loss诊断手册 | ✅ 已完成 |
| P6 DPO β ablation | ✅ 已完成 |
| PoC SFT（500条，r=16） | ✅ 已完成 |
| PoC DPO（70对，pipeline验证） | ✅ 已完成 |
| 正式SFT（~2万条，r=64） | ✅ 已完成 |
| 正式DPO（~2,000对，hard-negative） | ✅ 已完成 |
| Golden 回归评测 pipeline | ✅ 已完成 |
| vLLM 部署 + 压测 | ✅ 已完成 |
| RAG 链路 + 评测 | ✅ 已完成 |
| 轻量 Agent（function calling + trace） | ✅ 已完成（缺口已定位） |

---

## 1. 训练与对齐（核心主线）

### 1.1 数据工程设计（SFT 数据管线）

#### 1.1.1 数据来源与处理目标

- 原始数据来源：DISC-Law-SFT（约 9 万条法律问答）
- 当前用于统计的数据文件：`data/law_qa_alpaca.json`
- 目标：将"数据清洗"升级为"数据工程设计"，通过对照实验量化证明采样策略有效性，而非经验拍脑袋

**数据阶段分层说明（防止混淆）：**

| 阶段 | 规模 | 用途 |
|------|------|------|
| 上游原始数据 | ~9万条 | DISC-Law-SFT 来源 |
| 候选池 pool | 67,261条 | P3对照实验共同数据源（基础清洗后） |
| P3 A/B 组 | 各500条 | 对照实验（A随机，B工程化）；B组同时是PoC SFT数据 |
| 正式SFT训练集 | ~2万条 | 正式训练（同策略更大规模） |
| PoC DPO偏好数据 | 70对 | 20手工+50自动，验证DPO pipeline |
| 正式DPO偏好数据 | ~2,000对 | hard-negative为主，正式对齐训练 |

#### 1.1.2 P1 数据统计看板（已完成）

统计脚本：`stats_dashboard.py`

```bash
python stats_dashboard.py
```

产出文件：
- `analysis/sft_data_stats.json`
- `analysis/category_distribution.csv`
- `analysis/sft_data_dashboard.md`

#### 1.1.3 统计结果（当前版本）

有效样本数：**483**

长度分布（字符级）：

| 指标 | input | output |
|------|------:|------:|
| mean | 55.58 | 330.86 |
| P50 | 48.0 | 298.0 |
| P95 | 127.9 | 739.2 |
| max | 309 | 1440 |

质量指标：

| 指标 | 数值 |
|------|-----:|
| output < 30 占比 | 0.0% |
| URL 噪声占比 | 0.0% |
| 法条引用占比 | 72.67% |
| 近重复(input)占比 | 0.0% |
| 近重复(output)占比 | 0.0% |

#### 1.1.4 当前工程结论（基于 P1）

1. 当前数据在短答、URL噪声、重复样本维度上已达到较高洁净度
2. 输出长度显著高于输入长度，符合"法律解释型回答"特征
3. 法条引用比例较高（72.67%），可支撑法律风格强化目标
4. 后续需结合类别分布和对照实验进一步评估偏置风险与泛化能力

#### 1.1.5 风险与后续计划

- [待补] 类别均衡分析（劳动/婚姻/合同等）
- [x] "随机 500 vs 工程化 500"对照实验（P3 已完成）
- [待补] 偏置缓解策略（重采样、补样本、阈值回调）

#### 1.1.6 清洗规则决策（P2，已完成）

为回答"阈值怎么来的、是否会误伤"，将清洗规则工程化为四元组文档（规则-阈值来源-命中比例-偏置风险）：

- 文档：`docs/data_rules_rationale.md`
- 统计产物：`analysis/rule_hits.json`、`analysis/rule_hits.md`

```bash
python advanced_clean.py
```

**P2 统计结果：**

| 阶段 | 样本数 |
|------|-------:|
| 原始样本 | 483 |
| 基础清洗后 | 483 |
| 质量过滤输入 | 483 |
| 质量过滤后 | 481 |
| 质量阶段删除量 | 2 |

规则命中明细：

| 规则Key | 规则 | 命中数 | 在质量阶段命中比例 | 对原始总量删除贡献 |
|---------|------|-------:|------------------:|------------------:|
| low_legal_relevance | no legal keywords in output | 2 | 0.41% | 0.41% |

**工程解读：**

1. 当前数据在长度、噪声、重复等维度已较洁净，质量规则未触发大量删除
2. 本轮过滤主要由"法律相关性兜底规则"生效，用于保证领域一致性
3. 该规则存在潜在误伤风险：可能删除"语义合法律但未显式出现关键词"的样本
4. 后续将对被删除样本进行人工抽检，并评估是否引入语义分类器替代纯关键词规则

**关键工程要点：** 把每条清洗规则拆成"规则-阈值来源-命中比例-偏置风险"四元组并输出结构化统计。质量阶段仅删除 0.41%，说明数据已较成熟，规则主要承担领域一致性兜底而不是粗暴筛除。

---

#### 1.1.7 P3 对照实验（随机500 vs 工程化500，已完成）

**实验目标：** 验证"数据工程策略"是否在同等训练配置下带来稳定收益，而非随机波动。

**候选池与采样：**
- 候选池文件：`raw_data/law_qa_pool.jsonl`
- 候选池规模：**67,261**（基础可用性清洗后的统一候选池；pool 只做字段归一和长度过滤，不做质量过滤——质量过滤是B组的核心策略，提前做了对照就不干净了）
- A组：随机采样 500（seed=42）
- B组：分层 + 质量评分采样 500（seed=42）
- 约束：A/B 使用同一候选池与同一训练范式，仅替换训练数据策略

> **重要说明：** B 组这 500 条数据有双重身份——既作为 P3 对照实验的工程化组验证采样策略，也同时作为 **PoC 阶段 SFT 的训练数据**验证 pipeline 可跑通。一数两用，不是两批独立数据。

**B组工程化采样规则：**

分层：用关键词规则将样本粗分到劳动、婚姻家庭、合同、交通事故、刑事、房产、公司商事、民事程序、行政等类别，按各类别在 pool 中的占比分配配额，确保采样分布与候选池一致。

质量评分（五信号）：

| 信号 | 权重 |
|------|------|
| 长度带（输入10~180，输出80~800） | +1 |
| 法条条款引用（"第X条"） | +1 |
| 法律名称引用（"《××法》"） | +0.8 |
| 结构化关键词（"建议/依据/应当/综上"等） | +0.8 |
| URL噪声惩罚（http/www） | -1 |

每个类别桶内按质量分排序取前 N 个，不足则用全局高分样本补齐，确保稳定产出 500 条。

**训练结果：**

| 组别 | eval_loss |
|------|----------:|
| A（随机500） | 1.1755 |
| B（工程化500） | 1.0844 |

相对下降（B vs A）：**7.75%**

**自动评测结果（20题，0-10）：**

| 维度 | A | B |
|------|--:|--:|
| law_basis | 1.90 | 2.30 |
| structure | 1.55 | 2.55 |
| coverage | 2.10 | 2.30 |
| length | 1.00 | 1.00 |
| total | 6.55 | 8.15 |

胜场统计：**A=2, B=15, Tie=3**（20题样本下二项检验 p≈0.002，统计显著）

**P3 结论：**

1. B 在 eval_loss 上优于 A，相对下降 7.75%
2. B 在自动评测总分上显著更高（8.15 vs 6.55）
3. 胜场统计明显倾向 B（15:2），与训练指标方向一致
4. seed 与候选池固定，收益可复现、可归因
5. 该结论作为正式训练采样策略的依据

---

### 1.2 SFT 训练方案

#### 1.2.1 训练目标与机制

- 目标函数：最大化 P(y|x)，最小化回答 token 的 NLL loss
- 默认策略：`train_on_prompt=false`（仅在 assistant token 上算 loss）
- 原因：聚焦回答质量学习，避免 prompt token 稀释监督信号

#### 1.2.2 两阶段参数配置

> **注意：SFT 分 PoC 和正式训练两套配置，不能混用。**

**PoC 阶段（500条，快速验证 pipeline）：**

| 参数 | 值 | 说明 |
|------|-----|------|
| 基座模型 | Qwen2.5-7B-Instruct | — |
| 微调方式 | QLoRA (NF4) + LoRA | 4-bit量化 + double quant |
| lora_rank | **16** | 快速验证，资源省 |
| lora_alpha | **32** | alpha=2r，缩放比稳定 |
| lora_dropout | 0.05 | 抑制过拟合 |
| learning_rate | 2e-4 | P4对照实验确定的最优值 |
| 有效batch | 64（micro=4, accum=16） | 梯度信噪比与显存的最优平衡 |
| epochs | 3 | — |
| cutoff_len | 2048 | 与推理侧对齐 |
| 目标 | 跑通pipeline，约30-40分钟 | — |

**正式训练阶段（~2万条，深度领域适配）：**

| 参数 | 值 | 说明 |
|------|-----|------|
| lora_rank | **64** | 捕捉更复杂法律推理模式 |
| lora_alpha | **128** | alpha=2r，保持缩放比 |
| 其余参数 | 与PoC一致 | — |
| 目标 | 领域深度适配，约4-6小时 | 4090上 |

**为什么有效 batch 选 64：**

1. **梯度信噪比**：LoRA 只训练 <1% 参数，小 batch 的梯度方差对低秩矩阵 A/B 的收敛影响被放大，batch=64 是该量级的工程甜点
2. **QLoRA 反量化开销摊销**：NF4→BF16 反量化是固定 per-token 开销，增大 micro_batch 可有效摊销，提升 MFU（micro=4 时约 55% vs micro=1 时约 40%）
3. **显存约束下的最优拼法**：4090 24GB 下 micro=4, accum=16 是显存利用率和收敛稳定性的最优平衡，峰值约 16GB

#### 1.2.3 数据格式与注册流程

1. 原始数据下载与清洗
2. 精选样本（工程化采样策略）
3. 格式转换（Alpaca / ShareGPT）
4. 放入 `data/`
5. 在 `dataset_info.json` 注册
6. 启动 LLaMA-Factory SFT 训练

#### 1.2.4 训练执行步骤

```bash
llamafactory-cli train <your_sft_yaml>
```

检查产物：
- `training_loss.png`
- `trainer_state.json` / `all_results.json`
- LoRA adapter 输出目录（`adapter_model.safetensors` 等）

#### 1.2.5 Template 一致性约束（关键机制）

训练 template 与推理 template 必须一致。不一致会导致：special token 排布变化 → 输入拼接模式变化 → 分布偏移（distribution shift）。本项目默认使用：`template: qwen`。

#### 1.2.6 PoC SFT 实验结果（真实记录）

| 指标 | 数值 |
|------|-----:|
| epoch | 2.9217 |
| train_loss | 1.1283 |
| eval_loss | 1.1833 |
| train_runtime(s) | 2508.8263 |
| train_samples_per_second | 0.519 |

#### 1.2.7 正式 SFT 实验结果（~2万条，r=64）

| 指标 | 数值 |
|------|-----:|
| epoch | 2.9841 |
| train_loss | 0.8973 |
| eval_loss | 0.9312 |
| train_runtime(s) | 18742.3 |
| train_samples_per_second | 3.187 |

相比 PoC 阶段，正式训练 eval_loss 从 1.1833 降至 0.9312，相对下降 **21.3%**，符合数据规模扩大（500→2万）带来的预期收益。

#### 1.2.8 结果解读

1. 训练已稳定跑通，模型可正常推理，无致命异常
2. eval_loss 与 train_loss 接近，未出现明显过拟合信号
3. 对话对比显示：SFT 后回答更精炼、法条引用更具体、咨询风格更专业
4. 正式训练（~2万条，r=64）相较 PoC（500条，r=16）eval_loss 进一步下降 21.3%，深度领域适配效果显著

#### 1.2.9 SFT 能力边界定义

本项目 SFT 目标不是注入完整法律知识体系，而是：
- 强化法律回答风格
- 提升法条引用精确度
- 改善结构化表达

知识扩展与偏好优化主要依赖：DPO 对齐 + RAG 检索增强

---

### 1.3 DPO 偏好对齐

#### 1.3.1 目标

在 SFT 基础上进一步优化"偏好层面质量"：法条引用准确性、回答完整度、拒答边界与实操路径清晰度。

#### 1.3.2 两阶段策略与数据构造

> **注意：DPO 分 PoC（70对）和正式训练（约2,000对）两阶段，两阶段构造方式不同。**

**第一阶段：PoC 验证（70对）**

目的：验证 DPO pipeline 稳定、reference/β 机制正常工作、回归门禁能及时捕捉副作用；同时建立"小数据DPO"规模基线。

构造方式（两路合并）：
- **手工高质量**：约 20 对，覆盖"漏引法条"、"漏关键要件"、"程序性错误"等关键风险题型（自动构造的盲区）。chosen 是完整包含法律依据与操作建议的版本；rejected 则更模糊、缺少明确条款或关键步骤
- **自动构造**：50 对（seed=42），从 SFT 数据里采样同一问题，用两种不同采样温度（temperature=0.3 vs 0.8）生成两个回答，再用评分规则判断 chosen/rejected；分数差 △≥1.5 才纳入，保证区分度

**第二阶段：正式训练（约2,000对）**

目的：在 PoC 验证 pipeline 稳定后，通过更大规模和更高质量的偏好数据实现正式对齐。

构造方式：以 **hard-negative 为主**——rejected 不再是简单截断，而是"法条引用正确但漏关键要件/程序"的版本，迫使模型学到更精细的法律推理边界，避免 rejected 过弱带来的训练无效或过拟合。

两阶段均做格式校验：每条必须同时有 instruction/input/chosen/rejected，强约束 chosen 比 rejected 更详细。

#### 1.3.3 训练配置

| 参数 | PoC（70对） | 正式训练（~2,000对） |
|------|------------|---------------------|
| learning_rate | 5e-5 | 5e-5 |
| pref_beta | 0.1 | 0.1（扫0.05/0.1/0.2） |
| 有效batch | 16（micro=2, accum=8） | 32（micro=2, accum=16） |
| epoch | 1 | 1-2 |
| pref_loss | sigmoid | sigmoid |

**为什么 DPO 学习率（5e-5）比 SFT（2e-4）低：** DPO loss 的 landscape 比 SFT 更非线性，学习率过大容易让 chosen/rejected 差距过快拉大，触发 accuracies 过快到 1.0（过拟合偏好对），还会导致远离 reference policy 输出质量崩塌。

**为什么 DPO 有效 batch 比 SFT 小：** DPO 训练时每个 batch 需要同时处理 chosen 和 rejected 两条序列（等效 SFT 的 batch 翻倍），加之 reference model 的前向推理，显存压力更大。

#### 1.3.4 机制解释

- DPO 直接优化"chosen 相对 rejected 的偏好概率差"，不需要单独训练奖励模型
- reference model（训练全程冻结，只做前向推理不参与反向传播）的作用：稳定更新、约束偏移，避免策略过度偏离
- β 参数控制偏离 reference 的惩罚强度：β 大更保守更稳，β 小更激进收益可能更高但风险更大
- **β 选择标准**：不看 DPO loss，看 Golden 回归——law_accuracy/coverage 有提升且 repetition/avg_len/hallucination 不恶化，才算 β 选对了

#### 1.3.5 PoC 训练结果

| 指标 | 数值 |
|------|-----:|
| epoch | 2.6154 |
| train_loss | 0.1079 |
| eval_loss | 0.0976 |
| eval_rewards/margins | 4.4497 |
| eval_rewards/accuracies | 1.0000 |
| eval_rewards/chosen | 5.7655 |
| eval_rewards/rejected | 1.3158 |
| train_runtime(s) | 567.7126 |

#### 1.3.6 正式 DPO 训练结果（~2,000对，r=64，β=0.1）

| 指标 | 数值 |
|------|-----:|
| epoch | 1.9823 |
| train_loss | 0.3241 |
| eval_loss | 0.3587 |
| eval_rewards/margins | 3.2814 |
| eval_rewards/accuracies | 0.8916 |
| eval_rewards/chosen | 4.1253 |
| eval_rewards/rejected | 0.8439 |
| train_runtime(s) | 4821.3 |

与 PoC 相比，正式 DPO 的 accuracies 从 1.0 回落至 0.8916，说明 hard-negative 构造的 rejected 有效提升了训练难度，模型未出现过拟合偏好对的情况；eval_loss 0.3587 与 train_loss 0.3241 接近，泛化稳定。

#### 1.3.7 结果解读

1. DPO 训练流程已稳定跑通，loss 明显下降
2. **PoC accuracies=1.0** 在小数据（70对）场景下是预期内的：PoC 的目的是验证 pipeline 工作，不是追泛化效果；accuracies 过快到 1.0 是监控信号，提示 rejected 可能过弱——这正是正式训练阶段换用 hard-negative 的动因
3. **正式 DPO accuracies=0.8916** 说明 hard-negative 策略有效，模型在更难的偏好对上仍能学到方向，但未过拟合
4. 三方对比中，DPO 在法条引用准确性、回答完整度上优于 SFT

---

### 1.4 Ablation 与稳定性分析

#### 1.4.1 P4：Learning Rate 小型对照实验（已完成）

在同一数据与同一训练配置下，仅改变 learning rate 做 3 组快速对照：

| learning rate | train_loss | eval_loss | train_runtime(s) | 结论 |
|--------------:|----------:|----------:|-----------------:|------|
| 5e-5 | 1.1399 | 1.1025 | 77.8442 | 收敛偏慢，效果最弱 |
| 1e-4 | 1.1219 | 1.0824 | 77.8653 | 中等表现 |
| **2e-4** | **1.1087** | **1.0682** | 77.8786 | **最优** |

**P4 结论：** 后续 SFT 默认学习率采用 **2e-4**。

#### 1.4.2 P5：Loss 诊断手册（已完成）

文档：`docs/loss_diagnosis_manual.md`

排查优先级：**lr → batch → data → rank**

| 现象 | 典型信号 | 优先排查 | 动作示例 |
|------|----------|----------|----------|
| 震荡 | train loss 上下波动大 | lr / 有效batch | lr 下调 2x；增大 grad_accum |
| 发散 | loss 持续上升、NaN/Inf | lr / 数值稳定性 | lr 降到 1/2~1/4；开启 grad clip |
| 过拟合 | train 降、eval 升 | data / 正则 / rank | early stop；dropout↑；rank 16→8 |
| 欠拟合 | train、eval 都高且降不动 | lr / 训练步数 / rank | lr 上调；epoch↑；rank 8→16 |

#### 1.4.3 P6：DPO β Ablation（0.05 / 0.1 / 0.2，已完成）

| pref_beta | train_loss | eval_loss | eval_rewards/accuracies | eval_rewards/margins | 结论 |
|----------:|----------:|----------:|------------------------:|---------------------:|------|
| 0.05 | 0.1451 | 0.1048 | 1.0000 | 2.8953 | 稳定，但偏好强度中等 |
| **0.10** | **0.1079** | **0.0976** | **1.0000** | 4.4497 | **最优（当前默认）** |
| 0.20 | 0.1313 | 0.2145 | 0.8333 | 6.3224 | 过激，泛化变差风险高 |

**β 选择依据：** beta=0.1 取得最低 eval_loss 且 accuracies=1.0，综合最优；beta=0.2 虽然 margin 更大，但 eval_loss 恶化、accuracies 下降至 0.8333，过激不作默认配置。后续 DPO 默认采用 **pref_beta=0.1**。

#### 1.4.4 小结

- **超参选择**：通过 P4 对照采用 `learning_rate=2e-4`（SFT），通过 P6 对照采用 `pref_beta=0.1`（DPO）
- **诊断能力**：通过 P5 手册与脚本，形成"现象→动作→证据"标准化排查闭环，而非经验式试错

---

### 1.5 灾难性遗忘与能力边界

#### 1.5.1 评估目标

1. 法律能力是否提升？
2. 通用能力是否明显退化（灾难性遗忘）？

#### 1.5.2 通用能力评测结果（ceval）

| 维度 | Base | SFT | DPO | SFT相对Base | DPO相对Base |
|------|-----:|----:|----:|------------:|------------:|
| Average | 78.83 | **79.42** | 79.05 | +0.59 | +0.22 |
| STEM | 72.79 | 73.26 | **73.72** | +0.47 | +0.93 |
| Social Sciences | 85.82 | **86.18** | 85.82 | +0.36 | 0.00 |
| Humanities | 78.21 | **79.38** | 78.60 | +1.17 | +0.39 |
| Other | **80.99** | 81.51 | 80.47 | +0.52 | -0.52 |

**结论：** 未出现灾难性遗忘。SFT/DPO 后通用能力基本持平（Average 变化在 ±0.6 以内）。

#### 1.5.3 法律专项评测结果（10 case）

| 模型 | 平均关键点覆盖率 |
|------|----------------:|
| Base | 9.9% |
| SFT（PoC） | 13.3% |
| SFT（正式） | 17.8% |
| DPO（正式） | **22.4%** |

- 正式SFT 相对 Base：**+7.9 个百分点**
- 正式DPO 相对 正式SFT：**+4.6 个百分点**
- 正式DPO 相对 Base：**+12.5 个百分点**

#### 1.5.4 结论与原理

1. **通用能力侧**：Base→SFT→DPO 未出现明显退化，满足"可部署"底线
2. **法律能力侧**：正式SFT 已大幅提升法律风格与关键点覆盖，正式DPO 进一步提升准确性与完整度
3. **未明显遗忘的原因**：LoRA 仅训练极少量参数（<1%），基座权重冻结；基座模型（Qwen2.5-7B-Instruct）能力较强，领域数据对通用分布冲击有限

---

## 2. 标准化评测体系（核心主线）

### 2.1 评测目录结构与 Pipeline（已落地）

```
eval/
├── golden_cases.json       # 30题门禁集（分钟级高频跑）
├── auto_eval.py
├── metrics.py
├── regression_check.py
├── run_eval_pipeline.sh
├── runs/                   # 各次评测快照
└── reports/                # 自动生成报告
```

**评测逻辑：**
- 门禁集：30题，覆盖核心题型，分钟级出结果，每次训练后高频跑
- 扩展集：100+题，覆盖边缘case，weekly低频跑
- 两套集合职责不同：门禁集追快速可信告警，扩展集追全面覆盖

标准流程：
1. 固定 Golden 用例集（同题同口径）
2. 对 `Base / SFT / DPO` 生成答案（固定推理参数，`do_sample=false`）
3. 计算统一五维指标
4. 三个 summary 差分对比（Base→SFT 检查SFT副作用，SFT→DPO 检查DPO副作用）
5. 产出回归报告并触发告警

```bash
bash eval/run_eval_pipeline.sh
```

### 2.2 指标体系（已实现）

| 指标 | 定义 | 目标 | 门禁阈值 |
|------|------|------|----------|
| law_accuracy | 法律依据点命中比例 | 越高越好 | 下降 >2pct 报警 |
| coverage | gold key points 命中比例 | 越高越好 | 下降 >2pct 报警 |
| repetition_rate | 回答内部重复程度 | 越低越好 | 上升 >5pct 报警 |
| avg_len | 回答长度（字符近似） | 辅助诊断 | 不独立报警，联动分析 |
| hallucination_count | 命中禁用断言数 | 越低越好 | 增加 >2 报警 |

> **avg_len 使用说明：** avg_len 本身不单独做门禁，而是诊断辅助指标。当 coverage 报警时，联动看 avg_len：若 avg_len 也下降，说明"输出变短导致漏点"（修复方向：加 prompt 约束）；若 avg_len 正常，说明"输出够长但关键信息缺失"（修复方向：DPO 补 hard-negative）。两种情况根因不同，修复路径不同。

### 2.3 回归评测结果

#### PoC 阶段回归结果（run_20260426_131632）

| Model | law_accuracy | coverage | repetition_rate | avg_len | hallucination_count |
|-------|-------------:|---------:|----------------:|--------:|--------------------:|
| base | 0.850 | 0.433 | 0.000 | 545.0 | 0 |
| sft | 0.850 | 0.267 | 0.000 | 308.5 | 0 |
| dpo | 1.000 | 0.300 | 0.000 | 650.9 | 0 |

告警：`[ALERT] SFT vs BASE: coverage drop > 2pct`

**根因分析：**

1. SFT 输出变短（545→308.5），关键点漏答，直接拉低 coverage
2. hallucination_count=0、repetition=0，问题主要不在幻觉/重复，而在覆盖不足
3. 评测 key_points 偏字符串精确，对等价表达容错不足，可能低估真实覆盖
4. DPO 提升了法律依据准确性（1.000），但"多点展开覆盖"尚未充分恢复

#### 正式训练后回归结果（run_20260512_094817）

| Model | law_accuracy | coverage | repetition_rate | avg_len | hallucination_count |
|-------|-------------:|---------:|----------------:|--------:|--------------------:|
| base | 0.850 | 0.433 | 0.000 | 545.0 | 0 |
| sft_formal | 0.883 | 0.467 | 0.000 | 512.3 | 0 |
| dpo_formal | 1.000 | 0.500 | 0.000 | 623.7 | 0 |

告警：无告警触发，所有指标均在门禁阈值内。

**结果解读：**

1. 正式 SFT 修复了 PoC 阶段的 coverage 退步问题（0.267→0.467），训练数据中增加"分点覆盖要素"的 prompt 约束发挥了作用
2. 正式 DPO 进一步将 law_accuracy 推至 1.000，coverage 提升至 0.500，avg_len 合理回升
3. 无 hallucination、无 repetition，整体质量稳定
4. 本轮正式训练后评测体系实现了"自动评测—告警—根因定位—修复方案—验证收敛"完整闭环

**改进动作（已执行）：**

1. **评测侧**：引入 normalized match + 同义写法词典，同时输出 strict / normalized 两套覆盖率
2. **SFT 训练侧**：prompt 模板中明确"请分点覆盖要素+依据+维权路径"
3. **DPO 偏好侧**：正式训练 2,000 对中，chosen 加入"覆盖完整度"偏好信号；hard-negative 增加"法条对但漏关键点"的 rejected

> **本项目评测体系已实现"自动评测—告警—根因定位—修复方案"闭环**。PoC 阶段发现真实退化信号（SFT coverage 下滑），形成训练与评测双侧针对性改进方案，正式训练后验证修复有效。

---

## 3. 推理优化与部署（核心主线）

### 3.1 vLLM 部署架构（已完成）

**部署链路：**

```
Merged DPO Model (FP16) → vLLM Serve → OpenAI-Compatible API → 两段式压测 → 生产参数固化
```

**启动命令（推荐生产配置）：**

```bash
vllm serve /root/autodl-tmp/merged_models/law_qa_dpo \
  --served-model-name "law-expert-fp16" \
  --host 0.0.0.0 \
  --port 6006 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --max-num-seqs 256
```

**关键配置项说明：**

| 参数 | 值 | 选择依据 |
|------|-----|----------|
| dtype | float16 | 训练与推理口径一致，避免量化误差影响评测 |
| max-model-len | 2048 | 与训练 cutoff_len 对齐；法律QA实测平均prompt ~500 tokens，2048足够且KV压力可控 |
| gpu-memory-utilization | 0.90 | 吃满显存留 10% 余量防 OOM；7B FP16 约 14GB，0.90×24GB ≈ 21.6GB 给 vLLM |
| enable-prefix-caching | true | 固定 system prompt 命中率约 60%+，有边际收益 |
| max-num-seqs | 256 | 阶梯扫描 128/192/256 后确定的第一吞吐杠杆最优值 |

### 3.2 机制原理

#### KV Cache 为什么减少计算

自回归生成时，历史 token 的 K/V 不再重复计算，仅计算新 token 的 K/V，计算复杂度从"重复重算历史"转向"增量计算"，吞吐提升明显。

#### PagedAttention 与 Block Table 映射机制

vLLM 通过分页化 KV Cache 管理（非连续物理映射）降低碎片，支持动态请求插入与释放，是 Continuous Batching 的基础。

#### TTFT 抖动/暴涨的原因链路

```
并发上升 → KV Cache 压力上升 → 请求排队 → prefill 延后 → TTFT/P99 抖动或暴涨
```

实测观测：c=8 时 TTFT P99 从毫秒级突增到秒级（9.4s），属于典型资源争用拐点信号。

#### max_num_seqs 对 TPS 的影响

max_num_seqs 决定 Continuous Batching 的 batch 上限，直接影响 GPU 利用率：太小则 GPU 吃不满，TPS 低；太大则 KV cache 压力上升，P99 抖动、失败率风险。本项目中它是最敏感参数，在 128/192/256 阶梯扫描后确定最终取 **256**。

### 3.3 性能压测与调优（已完成）

**压测方法：**
- 工具：OpenAI 兼容脚本（阶梯并发）+ Locust（稳定性压测）
- 口径：固定模型、固定提示模板、固定 max_tokens，保证不同配置可比

#### 并发阶梯压测结果（FP16，max_tokens=128）

| 并发 | 请求数 | TPS (tokens/s) | TTFT P99 (ms) | Latency P99 (ms) | Latency Avg (ms) | Avg Tokens |
|-----:|------:|---------------:|--------------:|----------------:|-----------------:|-----------:|
| 1 | 20 | 48.58 | 39.5 | 2588.8 | 2598.8 | 126.2 |
| 4 | 20 | 161.73 | 286.3 | 3295.1 | 3119.8 | 126.2 |
| 8 | 20 | 260.73 | 319.5 | 3413.2 | 3256.3 | 126.2 |
| **16** | **40** | **486.64** | **358.3** | **3727.6** | **3512.1** | **126.3** |

**长输出场景（FP16，c=8，max_tokens=256）：**
- TPS：263.04，TTFT P99：440.8ms，Latency P99：6692.0ms，Avg Tokens：244.4

#### 关键参数调优案例

| 参数 | 对比 | 结论 |
|------|------|------|
| gpu-memory-utilization | 0.85 vs 0.95 | TPS 基本不变，在显存充裕场景影响有限 |
| max-model-len | 2048 vs 4096 | TPS/P99 基本无差异（当前业务输入远低于上限） |
| enable-prefix-caching | 关 vs 开 | 冷启动 TTFT 74.2ms → 命中后 67.0ms（-9.7%） |
| **max_num_seqs** | **阶梯扫描128/192/256** | **最关键杠杆，直接决定吞吐上限，最终取256** |

#### 稳定性压测（Locust，60秒持续）

- 5/10/20 用户失败率均为 **0%**
- P99 延迟线性增长，无崩溃
- 20 并发下服务稳定，满足当前 SLA 目标

**调优优先级：** max_num_seqs > enable-prefix-caching > gpu-memory-utilization > max-model-len

---

## 4. RAG 增强（辅助主线）

> 定位：RAG 是对 SFT/DPO 的外部知识补充层，不替代微调主线。
> 目标：在模型"会说"的基础上，让回答"有依据、可追溯、可更新"。

### 4.1 检索链路设计

**Dense 检索方案（当前主线 V3）：**

```
离线：法律文本清洗切块 → bge-base-zh Embedding → FAISS 索引
在线：用户问题向量化 → top-k=3 检索 → 组装增强 Prompt → vLLM 生成
```

代码结构：
- `rag/01_prepare_corpus.py`
- `rag/02_build_index.py`
- `rag/03_retriever.py`
- `rag/04_rag_pipeline.py`
- `rag/05_evaluate_rag.py`

**切块策略对比：**

| 策略 | 特点 | 结果 |
|------|------|------|
| 按条款切分（regex） | 法条边界清晰，但长度波动大 | 中等 |
| 固定字数切分 | 实现简单，易破坏法律语义 | 最差 |
| **递归语义切分（推荐）** | 条款优先，段落/句子兜底 | **最优（Precision@3: 81.2%）** |

当前推荐参数：`chunk_size=300，chunk_overlap=50，top_k=3`

**配置选择依据：**
- chunk=300 是法律条款自然粒度（一个完整条款约200-400字）的中间值，overlap=50 防止条款在边界被截断
- top_k=3 是召回覆盖率和噪声引入的平衡点，实测 top_k=5 时误召回率明显上升

**链路演化：**
- V2（基础）：bge-small + FAISS
- V3（当前最佳）：优化切分 + bge-base Dense
- V4/V5（探索）：Hybrid(BM25+Dense)+RRF+Reranker（进行中）

### 4.2 评测结论与核心发现

**关键量化结论：**

| 场景 | 分数变化 | 根因 |
|------|----------|------|
| TC001（准确召回劳动合同法） | **+13.7** | 正确注入关键条款，模型有依据 |
| TC010（补充工伤条例） | **+11.0** | 外部知识补全了参数记忆边界 |
| TC003（误召回） | **-16.7** | 错证据注入，模型高置信自洽错误 |
| TC009（名誉权问题误召回劳动法） | **-14.7** | 词汇陷阱导致跨领域误召回 |

**核心结论：简单 RAG 平均 +0.9，但误召回导致最差 -16.7。**

- RAG 不是天然增益，是"证据注入系统"，证据质量决定上限
- **错证据比没证据更危险**：没有证据时模型更谨慎甚至拒答；错证据会让模型高置信自洽错误
- Dense 检索存在"词汇陷阱"：查询含"公司/同事"时，劳动法文档被错误拉近

**止损策略（优先级顺序）：**

1. **相似度阈值过滤**：低相似度不注入，宁可让模型更谨慎（阈值用退步样例集 calibration，目标"退步案例数减少≥60%且正向收益损失≤20%"）
2. **控制 top-k=3**：防止噪声扩散
3. **触发澄清/拒答**：低置信或冲突时不强行回答
4. **元数据过滤（law_type）**：先做类别预过滤，减少跨领域误召回

**与 SFT/DPO 的协同边界：**

- SFT/DPO 负责：回答风格、结构化表达、偏好对齐（"怎么说"）
- RAG 负责：外部知识注入、可追溯依据（"说什么依据"）
- RAG 上限取决于检索精度，而不是生成模型本身

### 4.3 当前阶段结论

1. 简单 RAG 相比纯微调平均总分有提升（+0.9），但收益不稳定
2. V3（Dense + bge-base + 递归语义切分）综合最优（准确率/延迟平衡）
3. RAG 的主要瓶颈是"召回质量"，不是"生成能力"
4. 后续优化优先级：**检索精度 > 生成调参**（先解决误召回，再讨论生成风格）

---

## 5. 轻量级 Agent（可选加分）

### 5.1 目标与边界

- 目标：验证 `LLM + Tool` 架构理解与工程落地能力；将 Agent 问题转化为可观测、可量化的工程问题
- 边界：仅做单工具 function calling，不实现复杂 Planner/多工具编排

### 5.2 最小实现

**工具：`search_law(query, top_k)`**

从本地法律向量库检索 Top-K 条法条片段。实现文件：`agent_minimal/search_tool.py`

**Function Calling 流程：**

```
用户问题 → LLM 判断是否调用 search_law → 执行工具检索 → 
工具结果注入对话上下文 → LLM 输出最终回答（结论+法律依据+维权建议）
```

**自动验证项（三条件全满足才算通过）：**

1. 是否触发 tool_call
2. 是否执行工具并返回 tool_result
3. 是否产出 final_answer

**当前验证结果：通过率 0/3**

> **这不是失败，而是量化的缺口。** 0/3 说明"工具触发失败"是当前主要缺口，问题不是"感觉 Agent 不聪明"，而是闭环链路的具体环节没打通——可能是工具选择策略、参数格式、或结果注入模板。

**修复路径（已规划）：**

1. **强约束触发**：在 system prompt 中明确 JSON schema 和工具触发条件
2. **固定注入模板**：确保 tool_result 以固定格式被模型正确消费
3. **回归验证**：用同批用例验证通过率提升

> 体现的工程化能力：把不可控的 Agent 问题变成可观测、可验证、可迭代的指标问题。

---

## 6. 实验复现指南

### 6.1 环境准备

```bash
cd ~/autodl-tmp/LLaMA-Factory
nvidia-smi
which llamafactory-cli
```

### 6.2 数据准备

1. 准备清洗后法律问答数据（Alpaca/ShareGPT 任一）
2. 将数据放入 `data/`
3. 在 `dataset_info.json` 注册数据集名称与字段映射
4. 用小样本先做一次 dry-run，确认 dataset 可被正确读取

### 6.3 SFT 训练

**PoC 阶段配置（快速验证 pipeline）：**

```yaml
model_name_or_path: Qwen2.5-7B-Instruct
stage: sft
finetuning_type: lora
quantization_bit: 4          # NF4
template: qwen
lora_rank: 16                # PoC阶段
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 2e-4          # P4实验确定
per_device_train_batch_size: 4
gradient_accumulation_steps: 16  # 有效batch=64
num_train_epochs: 3
cutoff_len: 2048
bf16: true
max_grad_norm: 1.0
```

**正式训练阶段只需修改：**

```yaml
lora_rank: 64
lora_alpha: 128
# 其余参数与PoC一致
```

```bash
llamafactory-cli train my_configs/<your_sft_yaml>.yaml
```

**关键产物：**
- `adapter_config.json` / `adapter_model.safetensors`
- `training_loss.png`
- `all_results.json`（提取 train_loss / eval_loss / epoch）

**PoC 基准结果：** train_loss=1.1283，eval_loss=1.1833，epoch=2.9217

**正式训练结果：** train_loss=0.8973，eval_loss=0.9312，epoch=2.9841

### 6.4 DPO 训练

#### 6.4.1 前置条件

- 已完成 SFT，有可用 SFT adapter
- 已准备偏好数据（instruction, input, chosen, rejected）
- `dataset_info.json` 已注册并设置 `"ranking": true`

#### 6.4.2 PoC 阶段配置（70对，验证 pipeline）

```yaml
stage: dpo
finetuning_type: lora
create_new_adapter: true
learning_rate: 5e-5
pref_beta: 0.1               # P6实验确定
pref_loss: sigmoid
template: qwen
per_device_train_batch_size: 2
gradient_accumulation_steps: 8   # 有效batch=16
num_train_epochs: 1
```

**正式训练阶段修改：**

```yaml
gradient_accumulation_steps: 16  # 有效batch=32
num_train_epochs: 1~2
# pref_beta 在0.05/0.1/0.2扫描，以Golden回归为选择标准
```

```bash
llamafactory-cli train my_configs/<your_dpo_yaml>.yaml
```

**PoC 基准结果：** train_loss=0.1079，eval_loss=0.0976，rewards/margins=4.4497，accuracies=1.0

**正式训练结果：** train_loss=0.3241，eval_loss=0.3587，rewards/margins=3.2814，accuracies=0.8916

**判定标准：**
- 成功信号：loss 下降稳定，rewards/margins 持续为正
- 风险信号：accuracies 过快到 1.0（PoC 70对属正常，正式训练出现则需检查 rejected 质量）；输出重复或过度冗长

### 6.5 评测与回归测试

```bash
bash eval/run_eval_pipeline.sh
```

**正式训练后回归结果（run_20260512_094817）：**

| Model | law_accuracy | coverage | repetition_rate | avg_len | hallucination_count |
|-------|-------------:|---------:|----------------:|--------:|--------------------:|
| base | 0.850 | 0.433 | 0.000 | 545.0 | 0 |
| sft_formal | 0.883 | 0.467 | 0.000 | 512.3 | 0 |
| dpo_formal | 1.000 | 0.500 | 0.000 | 623.7 | 0 |

告警：无

### 6.6 部署与压测

**部署：**

```bash
vllm serve /root/autodl-tmp/merged_models/law_qa_dpo \
  --served-model-name "law-expert-fp16" \
  --host 0.0.0.0 \
  --port 6006 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --max-num-seqs 256
```

**并发压测结果（FP16，max_tokens=128）：**

| 并发 | TPS (tokens/s) | TTFT P99 (ms) | Latency P99 (ms) |
|-----:|---------------:|--------------:|-----------------:|
| 1 | 48.58 | 39.5 | 2588.8 |
| 4 | 161.73 | 286.3 | 3295.1 |
| 8 | 260.73 | 319.5 | 3413.2 |
| **16** | **486.64** | **358.3** | **3727.6** |

**稳定性压测（Locust，60s）：** 5/10/20 用户失败率均为 0%

---

## 附录 A：关键配置清单

### A.1 SFT 训练配置

```yaml
# PoC 阶段
model_name_or_path: Qwen2.5-7B-Instruct
stage: sft
finetuning_type: lora
quantization_bit: 4
template: qwen
lora_rank: 16          # 正式训练改为 64
lora_alpha: 32         # 正式训练改为 128
lora_dropout: 0.05
learning_rate: 2e-4
per_device_train_batch_size: 4
gradient_accumulation_steps: 16   # 有效batch=64
num_train_epochs: 3
cutoff_len: 2048
bf16: true
max_grad_norm: 1.0
```

### A.2 DPO 训练配置

```yaml
# PoC 阶段（70对）
stage: dpo
finetuning_type: lora
create_new_adapter: true
learning_rate: 5e-5
pref_beta: 0.1
pref_loss: sigmoid
template: qwen
per_device_train_batch_size: 2
gradient_accumulation_steps: 8    # 有效batch=16
num_train_epochs: 1
# 正式训练（~2,000对）：gradient_accumulation_steps改为16，epochs 1-2
```

### A.3 评测配置

- Golden 门禁集：`eval/golden_cases.json`（30题）
- 评测脚本：`eval/auto_eval.py`
- 回归告警：`eval/regression_check.py`
- 一键运行：`eval/run_eval_pipeline.sh`
- 核心阈值：coverage 下降 >2pct / law_accuracy 下降 >2pct / repetition_rate 上升 >5pct / hallucination_count 增加 >2

### A.4 推理部署配置（vLLM）

```bash
vllm serve /root/autodl-tmp/merged_models/law_qa_dpo \
  --served-model-name "law-expert-fp16" \
  --host 0.0.0.0 \
  --port 6006 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --max-num-seqs 256
```

### A.5 RAG 配置（当前 V3 最优）

- 检索：Dense（FAISS）
- Embedding：bge-base-zh
- 切分策略：递归语义切分
- chunk_size=300，chunk_overlap=50，top_k=3

### A.6 轻量 Agent 配置

- 工具：`search_law(query, top_k)`
- Agent 流程：function calling（最小实现）
- 轨迹日志：`agent_minimal/logs/trace_*.json`
- 当前状态：0/3（触发失败，缺口已定位，修复路径已规划）

---

## 附录 B：实验记录模板

### B.1 实验元信息

- 实验ID：
- 日期：
- 阶段：PoC / 正式训练 / Eval / Deploy / RAG / Agent
- 假设（Hypothesis）：

### B.2 输入配置

- 模型 + LoRA阶段（PoC r=16 / 正式 r=64）：
- 数据集（规模 + 来源）：
- 关键超参：
- 运行命令：

### B.3 指标结果

| 指标 | 值 | 备注 |
|------|--:|------|
| train_loss | | |
| eval_loss | | |
| coverage | | |
| law_accuracy | | |
| TTFT P99 | | |
| TPS | | |

### B.4 结果解读

1. 主要结论：
2. 与预期是否一致：
3. 异常点及根因：
4. 下一步动作：

---

## 附录 C：术语表

- **SFT (Supervised Fine-Tuning)**：监督微调，用标注问答对学习"如何回答"
- **DPO (Direct Preference Optimization)**：直接偏好优化，用 chosen/rejected 对学习"更偏好的回答"，无需单独训练奖励模型
- **LoRA / QLoRA**：低秩参数微调；QLoRA 在 NF4 量化基座上训练 LoRA 以省显存
- **PoC (Proof of Concept)**：小规模验证阶段（SFT 500条 r=16，DPO 70对），目标是跑通 pipeline
- **hard-negative**：正式 DPO 阶段的 rejected 构造方式——法条引用正确但漏关键要件/程序
- **train_on_prompt=false**：只在 assistant token 上计算损失
- **Catastrophic Forgetting**：领域微调后通用能力明显退化
- **RAG (Retrieval-Augmented Generation)**：检索增强生成，将外部知识注入上下文
- **Dense Retrieval**：基于向量语义相似度的检索
- **KV Cache**：缓存历史 token 的 Key/Value，减少重复计算
- **PagedAttention**：vLLM 的分页式 KV 管理机制，减少显存碎片
- **Continuous Batching**：vLLM 的动态批处理机制
- **max_num_seqs**：控制并行序列数的第一吞吐杠杆
- **TTFT (Time To First Token)**：首 token 延迟
- **TPS (Tokens Per Second)**：每秒生成 token 数（端到端，prompt+completion 均计入）
- **P99 延迟**：99 分位延迟，反映尾部请求体验
- **Function Calling**：模型按 schema 调用外部工具并回填结果
- **Golden 门禁集**：30题，高频跑，分钟级出结果；区别于低频跑的扩展集（100+题）
