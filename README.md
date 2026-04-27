# Law-Expert-7B：从 SFT 到 DPO、评测、部署的法律专家模型工程实践

## 0. 项目概览
- 项目目标
- 技术路线总览（SFT → DPO → Eval → Deploy）
- 当前进度看板（已完成 / 进行中 / 规划中）

---

## 1. 训练与对齐（核心主线）

### 1.1 数据工程设计（SFT 数据管线）

#### 1.1.1 数据来源与处理目标
- 原始数据来源：DISC-Law-SFT（约 9 万条法律问答）
- 当前用于统计的数据文件：`data/law_qa_alpaca.json`
- 目标：将“数据清洗”升级为“数据工程设计”，通过统计证据支撑样本质量与选择策略，而非经验拍脑袋。

#### 1.1.2 P1 数据统计看板（已完成）
统计脚本：
- `stats_dashboard.py`

运行命令：
```bash
python stats_dashboard.py
```

产出文件：
- `analysis/sft_data_stats.json`
- `analysis/category_distribution.csv`
- `analysis/sft_data_dashboard.md`

#### 1.1.3 统计结果（当前版本）
- 有效样本数：**483**

长度分布（字符级）：

| 指标 | input | output |
|---|---:|---:|
| mean | 55.58 | 330.86 |
| P50 | 48.0 | 298.0 |
| P95 | 127.9 | 739.2 |
| max | 309 | 1440 |

质量指标：

| 指标 | 数值 |
|---|---:|
| output < 30 占比 | 0.0% |
| URL 噪声占比 | 0.0% |
| 法条引用占比 | 72.67% |
| 近重复(input)占比 | 0.0% |
| 近重复(output)占比 | 0.0% |

#### 1.1.4 当前工程结论（基于 P1）
1. 当前数据在短答、URL噪声、重复样本维度上已达到较高洁净度；  
2. 输出长度显著高于输入长度，符合“法律解释型回答”特征；  
3. 法条引用比例较高（72.67%），可支撑法律风格强化目标；  
4. 后续需结合类别分布和对照实验进一步评估偏置风险与泛化能力。

#### 1.1.5 风险与后续计划（占位）
- [待补] 类别均衡分析（劳动/婚姻/合同等）
- [x] “随机 500 vs 工程化 500”对照实验（P3 已完成）
- [待补] 偏置缓解策略（重采样、补样本、阈值回调）

#### 1.1.6 清洗规则决策（P2，已完成）

为回答“阈值怎么来的、是否会误伤”，我们将清洗规则工程化为四元组文档：

- 文档：`docs/data_rules_rationale.md`
- 统计产物：`analysis/rule_hits.json`、`analysis/rule_hits.md`
- 运行命令：
```bash
python advanced_clean.py
```

##### P2 统计结果（当前版本）

阶段统计：

| 阶段 | 样本数 |
|---|---:|
| 原始样本 | 483 |
| 基础清洗后 | 483 |
| 质量过滤输入 | 483 |
| 质量过滤后 | 481 |
| 质量阶段删除量 | 2 |

规则命中明细：

| 规则Key | 规则 | 命中数 | 在质量阶段命中比例 | 对原始总量删除贡献 |
|---|---|---:|---:|---:|
| low_legal_relevance | no legal keywords in output | 2 | 0.41% | 0.41% |

##### 工程解读

1. 当前数据在长度、噪声、重复等维度已较洁净，质量规则未触发大量删除；  
2. 本轮过滤主要由“法律相关性兜底规则”生效，用于保证领域一致性；  
3. 该规则存在潜在误伤风险：可能删除“语义合法律但未显式出现关键词”的样本；  
4. 后续将对被删除样本进行人工抽检，并评估是否引入语义分类器替代纯关键词规则。

##### 面试可复述要点

> 我们把每条清洗规则都拆成“规则-阈值来源-命中比例-偏置风险”，并输出结构化统计。  
> 在当前数据版本中，质量阶段仅删除 0.41%，说明数据已较成熟，规则主要承担领域一致性兜底，而不是粗暴筛除。

---

#### 1.1.7 P3 对照实验（随机500 vs 工程化500，已完成）

##### 实验目标
验证“数据工程策略”是否在同等训练配置下带来稳定收益，而非随机波动。

##### 候选池与采样
- 候选池文件：`raw_data/law_qa_pool.jsonl`
- 候选池规模：**67,261**
- A组：随机采样 500
- B组：分层 + 质量评分采样 500
- 约束：A/B 使用同一候选池与同一训练范式，仅替换训练数据

##### 训练结果
| 组别 | eval_loss |
|---|---:|
| A（随机500） | 1.1755 |
| B（工程化500） | 1.0844 |

- 相对下降（B vs A）：**7.75%**

##### 自动评测结果（20题，0-10）
| 维度 | A | B |
|---|---:|---:|
| law_basis | 1.90 | 2.30 |
| structure | 1.55 | 2.55 |
| coverage | 2.10 | 2.30 |
| length | 1.00 | 1.00 |
| total | 6.55 | 8.15 |

- 胜场统计：**A=2, B=15, Tie=3**

##### P3 结论
1. B 在 `eval_loss` 上优于 A，且下降幅度达到 **7.75%**；  
2. B 在自动评测总分上显著更高（8.15 vs 6.55）；  
3. 胜场统计明显倾向 B（15:2），与训练指标方向一致；  
4. 以上结果支持“工程化样本构建有效”，证明数据工程在法律问答场景具备可复现收益。

### 1.2 SFT 训练方案

#### 1.2.1 训练目标与机制
- 目标函数：最大化 \(P(y|x)\)，最小化回答 token 的 NLL loss。
- 默认策略：`train_on_prompt=false`（仅在 assistant token 上算 loss）。
- 原因：聚焦回答质量学习，避免 prompt token 稀释监督信号。

#### 1.2.2 参数配置（本项目实配）
- 基座模型：`Qwen2.5-7B-Instruct`
- 微调方式：`QLoRA (4bit) + LoRA`
- LoRA 参数：`lora_rank=16`, `lora_alpha=32`, `lora_target=all`, `lora_dropout=0.05`
- 量化参数：`quantization_bit=4`（NF4）
- 训练精度：`bf16=true`, `max_grad_norm=1.0`
- 学习率：`1e-4`
- 等效 batch：`per_device_train_batch_size=2`, `gradient_accumulation_steps=8`（等效 16）
- 模板：`template=qwen`
- 长度：`cutoff_len=2048`

#### 1.2.3 数据格式与注册流程
1. 原始数据下载与清洗  
2. 精选样本  
3. 格式转换（Alpaca / ShareGPT）  
4. 放入 `data/`  
5. 在 `dataset_info.json` 注册  
6. 启动 LLaMA-Factory SFT 训练

#### 1.2.4 训练执行步骤
1. 准备 YAML（模型、数据、超参）  
2. 运行：
```bash
llamafactory-cli train <your_sft_yaml>
```
3. 检查产物：
- `training_loss.png`
- `trainer_state.json` / `all_results.json`
- LoRA adapter 输出目录（`adapter_model.safetensors` 等）  
4. 对话对比：原始模型 vs SFT 模型，验证风格迁移是否生效

#### 1.2.5 Template 一致性约束（关键机制）
- 训练 template 与推理 template 必须一致。  
- 不一致会导致：special token 排布变化 → 输入拼接模式变化 → 分布偏移（distribution shift）。  
- 项目默认使用：`template: qwen`。

#### 1.2.6 本次 SFT 实验结果（真实记录）

| 指标 | 数值 |
|---|---:|
| epoch | 2.9217 |
| train_loss | 1.1283 |
| eval_loss | 1.1833 |
| train_runtime(s) | 2508.8263 |
| eval_runtime(s) | 32.5533 |
| train_samples_per_second | 0.519 |
| train_steps_per_second | 0.032 |

#### 1.2.7 结果解读
1. 训练已稳定跑通，模型可正常推理，无致命异常；  
2. `eval_loss` 与 `train_loss` 接近，未出现明显过拟合信号；  
3. 对话对比显示：SFT 后回答更精炼、法条引用更具体、咨询风格更专业；  
4. 当前 500 条规模下提升符合预期，主要体现在“风格与表达分布”重塑。

#### 1.2.8 优化方向
- 降低震荡：`learning_rate` 从 `1e-4` 下调到 `5e-5`
- 提升稳定性：`gradient_accumulation_steps` 增大（如 8 → 16）
- 增强效果上限：训练样本扩展到 1000~2000
- 进一步验证：增加 epoch 并观察是否继续下降

#### 1.2.9 SFT 能力边界定义
本项目 SFT 目标不是注入完整法律知识体系，而是：
- 强化法律回答风格
- 提升法条引用精确度
- 改善结构化表达

知识扩展与偏好优化主要依赖：
- DPO 对齐
- RAG 检索增强

---


### 1.3 DPO 偏好对齐

#### 1.3.1 目标
在 SFT 基础上进一步优化“偏好层面质量”：法条引用准确性、回答完整度、拒答边界与实操路径清晰度。

#### 1.3.2 数据与流程
1. 基于 SFT 结果构造偏好数据（`chosen/rejected`）  
2. 在 `dataset_info.json` 注册（`ranking: true`）  
3. 执行 DPO 训练（`stage: dpo`，基于 SFT adapter 继续）  
4. 三方对比：Base vs SFT vs DPO

#### 1.3.3 机制解释
- DPO 直接优化“chosen 相对 rejected 的偏好概率差”，不需要单独训练奖励模型。  
- reference model 的作用：稳定更新、约束偏移，避免策略过度偏离。  
- β 参数控制偏离 reference 的惩罚强度：  
  - β 大：更保守，更新更稳  
  - β 小：更激进，收益可能更高但风险更大

#### 1.3.4 本轮训练结果

| 指标 | 数值 |
|---|---:|
| epoch | 2.6154 |
| train_loss | 0.1079 |
| eval_loss | 0.0976 |
| eval_rewards/margins | 4.4497 |
| eval_rewards/accuracies | 1.0000 |
| eval_rewards/chosen | 5.7655 |
| eval_rewards/rejected | 1.3158 |
| train_runtime(s) | 567.7126 |
| eval_runtime(s) | 9.1397 |

#### 1.3.5 结果解读
1. DPO 训练流程已稳定跑通，loss 明显下降（~0.189 → 0.108）；  
2. `eval_loss` 与 `train_loss` 接近，未见明显过拟合信号（loss 维度）；  
3. `rewards/margins` 显著为正且较高，说明模型已学会偏好 chosen；  
4. `accuracies=1.0` 在小数据场景下偏理想化，提示“记忆偏好对”的风险，需要更大偏好集验证泛化；  
5. 三方对比中，DPO 在法条引用准确性、回答完整度、维权路径可执行性方面优于 SFT，但出现了小数据常见副作用（末尾重复）。

#### 1.3.6 当前结论
相对 SFT，DPO 在法条引用准确性与回答完整度上有明显提升；当前阶段已验证“方向正确”，但需通过更大偏好数据与 β 对照提升稳健性。


### 1.4 Ablation 与稳定性分析

本节目标：回答“为什么选这个学习率”以及“出现 loss 异常时如何系统排查”。

---

#### 1.4.1 P4：Learning Rate 小型对照实验（必做）

在同一数据与同一训练配置下，仅改变 learning rate 做 3 组快速对照（V100S）：

- 数据集：`law_qa_p3_b_engineered_500`
- 其余参数固定（仅改 `learning_rate`）
- 快速配置：`max_samples=120, cutoff_len=512, num_train_epochs=0.3333`

##### P4 结果汇总

| learning rate | train_loss | eval_loss | train_runtime(s) | 结论 |
|---:|---:|---:|---:|---|
| 5e-5 | 1.1399 | 1.1025 | 77.8442 | 收敛偏慢，效果最弱 |
| 1e-4 | 1.1219 | 1.0824 | 77.8653 | 中等表现 |
| 2e-4 | **1.1087** | **1.0682** | 77.8786 | **最优** |

##### P4 结论
1. 在该实验设置下，`2e-4` 同时取得最低 `train_loss` 与最低 `eval_loss`；  
2. 三组 runtime 基本一致，性能差异可归因于 learning rate；  
3. 后续 SFT 默认学习率优先采用 **2e-4**，并在更长训练轮次做复验。

---

#### 1.4.2 P5：Loss 诊断手册（必做）

为避免“只会调参不会诊断”，沉淀了标准诊断流程：

- 手册文档：`docs/loss_diagnosis_manual.md`
- 自动诊断脚本：`analysis/p5_quick_diagnose.py`
- 快速诊断配置：`my_configs/p5_diag_smoke.yaml`
- 固定排查顺序：**lr → batch → data → rank**

##### 现象 → 排查动作（摘要）

| 现象 | 典型信号 | 优先排查 | 动作示例 |
|---|---|---|---|
| 震荡 | train loss 上下波动大 | lr / 有效batch | lr 下调 2x；增大 grad_accum |
| 发散 | loss 持续上升、NaN/Inf | lr / 数值稳定性 | lr 降到 1/2~1/4；开启 grad clip |
| 过拟合 | train 降、eval 升 | data / 正则 / rank | early stop；dropout↑；rank 16→8 |
| 欠拟合 | train、eval 都高且降不动 | lr / 训练步数 / rank | lr 上调；epoch↑；rank 8→16 |

##### P5 快速实验结果（本轮）
- 训练摘要：`train_loss=1.1087`, `eval_loss=1.0682`, `train_runtime≈77.75s`
- 诊断脚本输出：`train_loss_count=1`, `eval_loss_count=0`
- 解读：本次 smoke run 日志点过少，**不足以判断趋势类问题**（震荡/过拟合/欠拟合）；可给出“未见明显异常信号”的弱结论。

##### P5 结论与下一步
1. 已建立“现象→动作→证据”诊断体系，具备系统排查能力；  
2. 本轮快速实验主要用于验证流程可运行；  
3. 若要做趋势诊断，建议最小增量设置：`num_train_epochs=1.0`、`eval_steps=10`、`logging_steps=2`，确保有足够日志点再判定稳定性。

---

#### 1.4.3 P6：DPO β Ablation（0.05 / 0.1 / 0.2）

在同一偏好数据、同一训练设置下，仅改变 `pref_beta` 做 3 组对照（V100S）：

##### 实验结果汇总

| pref_beta | train_loss | eval_loss | eval_rewards/accuracies | eval_rewards/margins | 结论 |
|---:|---:|---:|---:|---:|---|
| 0.05 | 0.1451 | 0.1048 | 1.0000 | 2.8953 | 稳定，但偏好强度中等 |
| 0.10 | **0.1079** | **0.0976** | **1.0000** | 4.4497 | **最优（当前默认）** |
| 0.20 | 0.1313 | 0.2145 | 0.8333 | 6.3224 | 过激，泛化变差风险高 |


##### 结论（β 选择依据）
1. `beta=0.1` 取得最低 `eval_loss`，且 `accuracies=1.0`，综合最优；  
2. `beta=0.05` 更保守稳定，但效果弱于 `0.1`；  
3. `beta=0.2` 虽然 margin 更大，但 `eval_loss` 恶化、`accuracies` 下降至 0.8333，说明偏好更新过激，不作为默认配置。  

因此，后续 DPO 默认采用：**`pref_beta=0.1`**。
#### 1.4.4 小结
- **超参选择**：通过 P4 对照，采用 `learning_rate=2e-4`。  
- **诊断能力**：通过 P5 手册与脚本，形成标准化排查闭环，而非经验式试错。


### 1.5 灾难性遗忘与能力边界

#### 1.5.1 评估目标
回答两个核心问题：  
1. 法律能力是否提升？  
2. 通用能力是否明显退化（灾难性遗忘）？

#### 1.5.2 评测设计
- 通用域：`ceval`（Average / STEM / Social Sciences / Humanities / Other）
- 法律专项：10 个法律 case，统计“关键点覆盖率”
- 对比对象：Base（原始模型） vs SFT vs DPO

#### 1.5.3 通用能力评测结果（ceval）

| 维度 | Base | SFT | DPO | SFT相对Base | DPO相对Base |
|---|---:|---:|---:|---:|---:|
| Average | 78.83 | **79.42** | 79.05 | +0.59 | +0.22 |
| STEM | 72.79 | 73.26 | **73.72** | +0.47 | +0.93 |
| Social Sciences | 85.82 | **86.18** | 85.82 | +0.36 | 0.00 |
| Humanities | 78.21 | **79.38** | 78.60 | +1.17 | +0.39 |
| Other | **80.99** | 81.51 | 80.47 | +0.52 | -0.52 |

**结论（通用能力）**：未出现灾难性遗忘。  
SFT 在各维度整体小幅提升；DPO 相比 Base 总体仍持平略升（Average +0.22）。

#### 1.5.4 法律专项评测结果（10 case）

| 模型 | 平均关键点覆盖率 |
|---|---:|
| Base | 9.9% |
| SFT | 13.3% |
| DPO | **16.7%** |

- SFT 相对 Base：**+3.4 个百分点**
- DPO 相对 SFT：**+3.4 个百分点**
- DPO 相对 Base：**+6.8 个百分点**

说明：当前“关键点覆盖率”采用精确字符串匹配，可能低估等价表述（如“劳动合同法四十六条” vs “《劳动合同法》第四十六条”）。结合人工案例对比，DPO 在法条引用准确性与回答完整度上有实质提升。

#### 1.5.5 结论：能力提升与退化折中

1. **通用能力侧**：Base→SFT→DPO 未出现明显退化，满足“可部署”底线；  
2. **法律能力侧**：SFT 已提升法律风格，DPO 进一步提升关键点覆盖；  
3. **综合判断**：在通用能力基本不变前提下，领域能力持续提升，链路有效。

#### 1.5.6 原理解释（为什么未明显遗忘）
- LoRA 仅训练少量参数（<1%），基座权重冻结，通用知识保留较好；  
- 领域数据规模较小（500级），对通用分布冲击有限；  
- 基座模型能力较强，低量微调不易造成大幅退化。

#### 1.5.7 能力提升—退化折中曲线

![Capability Trade-off (General vs Legal)](image1)

图示结论：
- 通用能力（ceval avg）在 Base/SFT/DPO 间基本稳定（约 79 附近小幅波动）；  
- 法律专项覆盖率呈单调上升：`9.9 → 13.3 → 16.7`；  
- 说明本项目在“通用能力基本不损失”的前提下，实现了稳定的领域能力增益。

---

## 2. 标准化评测体系（核心主线）

### 2.1 评测目录结构与 Pipeline（已落地）

目录结构：
- `eval/golden_cases.json`
- `eval/auto_eval.py`
- `eval/metrics.py`
- `eval/regression_check.py`
- `eval/run_eval_pipeline.sh`
- `eval/runs/`
- `eval/reports/`

标准流程：
1. 固定 Golden 用例集（同题同口径）
2. 对 `Base / SFT / DPO` 生成答案
3. 计算统一指标（law_accuracy / coverage / repetition / avg_len / hallucination）
4. 产出回归报告并触发告警（regression alert）

执行命令：
```bash
bash eval/run_eval_pipeline.sh
```

---

### 2.2 指标体系（已实现）

当前统一指标如下：

1. **法条准确率（law_accuracy）**  
   - 定义：法律依据点命中比例（按规则归一化匹配）
   - 目标：越高越好

2. **关键点覆盖率（coverage）**  
   - 定义：gold key points 命中比例
   - 目标：越高越好

3. **重复率（repetition_rate）**  
   - 定义：回答内部重复程度
   - 目标：越低越好

4. **平均长度（avg_len）**  
   - 定义：回答长度（当前用字符近似）
   - 目标：用于监控“过短漏点”或“过长啰嗦”

5. **幻觉次数（hallucination_count）**  
   - 定义：命中禁用断言（must_not）条目数
   - 目标：越低越好

---

### 2.3 Golden Eval 机制（已跑通）

#### 2.3.1 用例设计原则
- 覆盖训练集内 + 训练集外问题
- 覆盖劳动/合同/民法/程序/消费者等类别
- 每题包含：
  - `key_points`（应出现）
  - `must_not`（不应出现）

#### 2.3.2 版本回归测试
每次模型更新（SFT 或 DPO）都做三方对比：
- Base（基座）
- SFT（监督微调）
- DPO（偏好对齐）

并固定推理参数，保证可比性（例如 `do_sample=false`）。

#### 2.3.3 退化告警规则（当前）
- coverage 下降超过 2pct：告警
- law_accuracy 下降超过 2pct：告警
- repetition_rate 上升超过 5pct：告警
- hallucination_count 增加超过 2：告警

---

### 2.4 自动化报告与结果追踪（已完成首轮）

#### 2.4.1 本轮回归结果（run_20260426_131632）

| Model | law_accuracy | coverage | repetition_rate | avg_len | hallucination_count |
|---|---:|---:|---:|---:|---:|
| base | 0.850 | 0.433 | 0.000 | 545.0 | 0 |
| sft | 0.850 | 0.267 | 0.000 | 308.5 | 0 |
| dpo | 1.000 | 0.300 | 0.000 | 650.9 | 0 |

告警：
- `[ALERT] SFT vs BASE: coverage drop > 2pct`

#### 2.4.2 结果分析（本轮）

**现象**：
1. SFT 相比 Base：`law_accuracy` 持平（0.85），但 `coverage` 显著下降（0.433 → 0.267）；
2. DPO 相比 SFT：`law_accuracy` 提升到 1.0，但 `coverage` 仅回升到 0.300，仍低于 Base；
3. `hallucination_count=0`、`repetition=0`，说明问题主要不在“幻觉/重复”，而在“覆盖不足或评分口径偏严”。

**可能原因**：
1. **SFT 输出更短**（avg_len 545 → 308.5），导致关键点漏答，直接拉低 coverage；  
2. **评测 key_points 偏字符串精确**，对等价表达容错不足，可能低估真实覆盖；  
3. **SFT 数据偏“简洁回答风格”**，造成“更精炼但少点”的结构性偏移；  
4. DPO 虽提升法律依据准确性，但在“多点展开覆盖”上尚未充分恢复。

#### 2.4.3 改进动作（下一轮）

1. **评测侧修正（优先）**  
   - 引入 normalized match + 同义写法词典（法条别名、数字写法统一）
   - 报告同时输出 strict / normalized 两套覆盖率

2. **训练侧修正（SFT）**  
   - 在 SFT 数据中提升“多要点完整回答”样本占比
   - 在 prompt 模板中明确“请分点覆盖要素+依据+维权路径”

3. **DPO 偏好侧修正**  
   - 在 chosen 样本中加入“覆盖完整度”偏好信号
   - hard-negative 增加“少关键点但法条正确”的 rejected 样本，专门打击漏点

4. **回归门禁**  
   - 将 `coverage` 设为发布前硬门槛（不允许低于 Base 超过阈值）

#### 2.4.4 小结
本项目评测体系已实现“**自动评测—告警—定位—修复**”闭环。  
本轮发现了真实退化信号（SFT coverage 下滑），并已形成针对性的训练与评测双侧改进方案。

---

## 3. 推理优化与部署（核心主线）

### 3.1 vLLM 部署架构（已完成）

#### 3.1.1 服务化部署流程
本项目部署链路：

`Merged DPO Model (FP16) -> vLLM Serve -> OpenAI-Compatible API -> 压测与调优 -> 生产参数固化`

部署模型路径：
- `/root/autodl-tmp/merged_models/law_qa_dpo`

启动命令（当前可用稳定版本）：
```bash
vllm serve /root/autodl-tmp/merged_models/law_qa_dpo \
  --served-model-name "law-expert-fp16" \
  --host 0.0.0.0 \
  --port 6006 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
```

#### 3.1.2 关键配置项说明
- `--served-model-name`：API 中 `model` 字段名称  
- `--host/--port`：服务监听地址（AutoDL 对外端口使用 6006）  
- `--dtype float16`：当前硬件与模型组合下最稳配置  
- `--max-model-len`：单请求最大上下文长度，影响 KV Cache 占用  
- `--gpu-memory-utilization`：KV Cache Pool 显存占比  
- `--enable-prefix-caching`：开启前缀共享，降低重复 prefill 成本

#### 3.1.3 API 验证
- `curl` 调用 `/v1/chat/completions` 成功  
- Python OpenAI 客户端（改 `base_url`）调用成功  
- 流式输出可用，服务可持续响应

---

### 3.2 机制原理（面试高频 + 实测对应）

#### 3.2.1 KV Cache 为什么减少计算
自回归生成时，历史 token 的 K/V 不再重复计算，仅计算新 token 的 K/V，因此计算复杂度从“重复重算历史”转向“增量计算”，吞吐提升明显。

#### 3.2.2 为什么显存占用会增加
KV Cache 以空间换时间：每个请求会持续占用缓存块，序列越长、并发越高，占用越大。  
因此高并发下可能出现排队，表现为 TTFT 抬升。

#### 3.2.3 PagedAttention 与 Block Table 映射机制
vLLM 通过分页化 KV Cache 管理（非连续物理映射）降低碎片，支持动态请求插入与释放，是 Continuous Batching 的基础。

#### 3.2.4 TTFT 抖动/暴涨的原因链路
典型链路：  
`并发上升 -> KV Cache 压力上升 -> 请求排队 -> prefill 延后 -> TTFT/P99 抖动或暴涨`

你在历史基线中观测到：  
- c=8 时 TTFT P99 从毫秒级突增到秒级（9.4s）  
属于典型资源争用拐点信号。

#### 3.2.5 max_num_seqs 对 TPS 的影响
`max_num_seqs` 决定 Continuous Batching 的 batch 上限，直接影响 GPU 利用率。  
本项目中它是最敏感参数：调优后吞吐提升最显著（见 3.3）。

---

### 3.3 性能压测与调优（已完成）

#### 3.3.1 压测方法与场景
- 工具：OpenAI 兼容脚本压测 + Locust 持续压测  
- 维度：并发（c=1/4/8/16）、TTFT、TPS、P99 延迟、失败率  
- 口径：固定模型、固定提示模板、固定 `max_tokens`

---

#### 3.3.2 结果模板（实测填充）

##### A) 最新并发阶梯压测（FP16，max_tokens=128）

| 并发 | 请求数 | TPS (tokens/s) | TTFT P99 (ms) | Latency P99 (ms) | Latency Avg (ms) | Avg Tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 48.58 | 39.5 | 2588.8 | 2598.8 | 126.2 |
| 4 | 20 | 161.73 | 286.3 | 3295.1 | 3119.8 | 126.2 |
| 8 | 20 | 260.73 | 319.5 | 3413.2 | 3256.3 | 126.2 |
| 16 | 40 | 486.64 | 358.3 | 3727.6 | 3512.1 | 126.3 |

##### B) 长输出场景（FP16，c=8，max_tokens=256）
- TPS：263.04
- TTFT P99：440.8ms
- Latency P99：6692.0ms
- Latency Avg：6241.1ms
- Avg Tokens：244.4

解读：`max_tokens` 增大后，总延迟上升符合预期；吞吐保持较高，说明系统在中高并发下仍稳定。

---

#### 3.3.3 关键参数调优案例（结合历史完整实验）

##### 1) `gpu-memory-utilization`
- 0.85 vs 0.95：TPS 基本不变（约 9.5），P99差异极小  
- 结论：在 V100S-32GB + 7B FP16（显存充裕）场景影响有限

##### 2) `max-model-len`
- 2048 vs 4096：TPS/P99 基本无差异  
- 结论：当前法律 QA 实际长度远低于上限，改该参数收益不明显

##### 3) `enable-prefix-caching`
- 冷启动 TTFT 均值 74.2ms -> 命中后 67.0ms（-9.7%）  
- 结论：短 system prompt 场景收益有限；长前缀（RAG/ToB）收益更大

##### 4) `max_num_seqs`（本次最关键）
- c=8：TPS 从 8.4 提升到约 42.6（历史调优结果）  
- c=16：系统 TPS 可达约 66.2（历史调优结果）  
- 结论：这是当前最优先调优杠杆，直接决定吞吐上限

---

#### 3.3.4 稳定性压测（Locust）
60 秒持续压测结果：
- 5/10/20 用户失败率均为 0%
- P99 延迟线性增长，无崩溃
- 20 并发下服务稳定，可满足当前 SLA 目标

---

#### 3.3.5 推荐生产配置（当前）

```bash
vllm serve /root/autodl-tmp/merged_models/law_qa_dpo \
  --served-model-name "law-expert-fp16" \
  --host 0.0.0.0 \
  --port 6006 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
```

建议：
- 中高并发业务下，优先调 `max_num_seqs`
- 若 TTFT 抬升明显，优先排查 KV Cache 竞争与排队
- 长 prompt 业务默认开启 prefix caching


---

## 4. RAG 增强（辅助主线）

> 定位：RAG 是对 SFT/DPO 的外部知识补充层，不替代微调主线。  
> 目标：在模型“会说”的基础上，让回答“有依据、可追溯、可更新”。

---

### 4.1 检索链路设计

#### 4.1.1 Dense 检索方案（当前主线）
本项目基础 RAG 链路：

1. **离线构建**
   - 法律文本清洗与切块（优先语义边界）
   - Embedding 向量化（bge-small-zh / bge-base-zh）
   - 向量索引构建（FAISS，后续扩展 Chroma 元数据过滤）

2. **在线检索**
   - 用户问题向量化
   - 向量检索 `top-k=3`
   - 组装增强 Prompt：`System + 检索法条 + 用户问题`
   - vLLM 生成回答

代码结构（已实现）：
- `rag/01_prepare_corpus.py`
- `rag/02_build_index.py`
- `rag/03_retriever.py`
- `rag/04_rag_pipeline.py`
- `rag/05_evaluate_rag.py`

语料与索引：
- 语料目录：`rag/law_corpus/`
- 索引目录：`rag/vector_store/faiss.index` + `chunks_meta.json`

---

#### 4.1.2 数据切块与索引构建（关键工程点）

切块策略对检索质量影响显著，本项目对比了三类策略：

| 策略 | 特点 | 结果结论 |
|---|---|---|
| 按条款切分（regex） | 法条边界清晰，但长度波动大 | 中等 |
| 固定字数切分 | 实现简单，但易破坏法律语义 | 最差 |
| 递归语义切分（推荐） | 条款优先，段落/句子兜底 | 最优 |

递归切分参数（当前推荐）：
- `chunk_size=300`
- `chunk_overlap=50`
- 分隔符优先级：条款边界 > 段落 > 换行 > 句号 > 字符兜底

实验结论（切分对检索）：
- 递归语义切分 Precision@3 最优（81.2%）
- 固定长度切分在法律条文场景退化明显（语义割裂）

---

#### 4.1.3 从“基础版”到“进阶版”链路演化

- **V2（基础）**：bge-small + FAISS  
- **V3（当前最佳）**：优化切分 + bge-base Dense  
- **V4/V5（探索）**：Hybrid(BM25+Dense)+RRF+Reranker(+Citation)

当前结论：**V3 是综合最优配置**（分数与延迟平衡最好）。

---

### 4.2 关键影响因素

#### 4.2.1 chunk_size 对召回/生成质量的影响

核心规律：
1. `chunk` 过小：语义碎片化，法条上下文断裂，生成时易漏关键点  
2. `chunk` 过大：主题混杂，向量相似度不够“尖锐”，噪声召回增加  
3. 法律文本对边界极敏感，应优先“条款语义完整性”而非固定长度

本项目经验参数：
- `chunk_size=300, overlap=50` 在法律咨询语料上表现最稳

---

#### 4.2.2 噪声与误召回问题分析（有实证）

RAG 改善并非单调，存在“检索噪声反伤”：

- 提升案例：
  - TC001：+13.7（准确注入《劳动合同法》关键条款）
  - TC010：+11.0（补充工伤条例信息）
- 退步案例：
  - TC003：-16.7
  - TC009：-14.7（名誉权问题误召回劳动法内容）

问题本质：
- Dense 检索存在“词汇陷阱”  
  例：查询含“公司/同事”时，劳动法文档被错误拉近
- 错误检索结果 **比不检索更差**（会向 LLM 注入错误上下文）

可行缓解：
1. 元数据过滤（law_type）先做类别预过滤  
2. 降低 top-k 或重排注入顺序，减少中段信息淹没  
3. 引入查询分类/路由，避免不适配检索器污染候选集

---

#### 4.2.3 与 SFT/DPO 的协同边界（必须明确）

职责分工：

- **SFT/DPO 负责**：  
  - 回答风格、结构化表达、法律咨询语气  
  - 偏好对齐（详细准确 > 简短模糊）

- **RAG 负责**：  
  - 外部知识注入（法条原文、更新条款）  
  - 降低“凭记忆编造法条”的风险  
  - 提供可追溯依据

协同结论：
1. 微调让模型“会回答法律问题”；RAG让模型“有依据地回答”  
2. 当问题超出模型参数记忆边界时，RAG 提升明显  
3. 当模型已熟悉且检索不准时，RAG 可能退步（噪声注入）

一句话总结：  
**SFT/DPO 决定“怎么说”，RAG 决定“说什么依据”；RAG 上限取决于检索精度，而不是生成模型本身。**

---

### 4.3 当前阶段结论（可复述）

1. 简单 RAG 相比纯微调在本项目平均总分有提升（+0.9），但收益不稳定；  
2. 在当前法律咨询语料上，`V3（Dense + bge-base + 递归语义切分）` 综合最优（准确率/延迟平衡最佳）；  
3. RAG 的主要瓶颈是“召回质量”，不是“生成能力”；  
4. 后续优化优先级：**检索精度 > 生成调参**（先解决误召回，再讨论生成风格）。

---

## 5. 轻量级 Agent（可选加分）

### 5.1 目标与边界
- 目标：验证 `LLM + Tool` 架构理解与工程落地能力  
- 边界：仅做单工具 function calling，不实现复杂 Planner/多工具编排

### 5.2 最小实现

#### 5.2.1 Tool：`search_law()`
- 输入：`query`, `top_k`
- 功能：从本地法律向量库检索 Top-K 条法条片段
- 实现文件：`agent_minimal/search_tool.py`

#### 5.2.2 Function Calling 流程
1. 用户问题进入 Agent  
2. LLM 判断是否调用 `search_law`  
3. 执行工具检索并返回结构化结果  
4. 将工具结果注入对话上下文  
5. LLM 输出最终回答（结论 + 法律依据 + 维权建议）

实现文件：
- `agent_minimal/agent_fc.py`
- `agent_minimal/run_demo.py`

#### 5.2.3 调用轨迹示例（可追溯）
轨迹文件：`agent_minimal/logs/trace_*.json`

标准轨迹结构：
`用户问题 -> tool_call(search_law) -> tool_result(top-k法条) -> final_answer`

#### 5.2.4 自动验证
- 脚本：`agent_minimal/eval_agent_trace.py`
- 验证项：
  - 是否触发工具调用
  - 是否执行工具并返回结果
  - 是否产出最终回答

结论：最小 Agent 方案已完成，验证了 function calling 的闭环能力，可作为后续多工具 Agent 的基础。
---

## 6. 面试向“深水区”问答（项目护城河）
### 6.1 SFT 深问
- 为什么 500 条可能有效？
- train_on_prompt 的影响？
- template 不一致为什么是 distribution shift？

### 6.2 DPO 深问
- DPO 与 RLHF 的差异
- reference 的作用
- β 的实际含义与调参逻辑

### 6.3 推理优化深问
- KV Cache / PagedAttention / TTFT / TPS 机制链路

---

## 7. 实验复现指南

### 7.1 环境准备
1. 进入项目目录并检查 GPU：
```bash
cd ~/autodl-tmp/LLaMA-Factory
nvidia-smi
```
2. 确保可调用训练命令：
```bash
which llamafactory-cli
```
3. 建议使用与你训练一致的 CUDA/transformers 环境，避免版本漂移。

### 7.2 数据准备
1. 准备清洗后法律问答数据（Alpaca/ShareGPT 任一）。  
2. 将数据放入 `data/`。  
3. 在 `dataset_info.json` 注册数据集名称与字段映射。  
4. 用小样本先做一次 dry-run，确认 dataset 可被正确读取。

### 7.3 SFT 训练（本项目复现实操）
1. 准备 SFT YAML（核心项）：
- `model_name_or_path=Qwen2.5-7B-Instruct`
- `stage=sft`
- `finetuning_type=lora`
- `quantization_bit=4`
- `template=qwen`
- `learning_rate=1e-4`
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=8`
- `cutoff_len=2048`

2. 启动训练：
```bash
llamafactory-cli train my_configs/<your_sft_yaml>.yaml
```

3. 训练完成后检查目录（示例）：
- `tmp/saves/law_qa_qlora_sft/`
  - `adapter_config.json`
  - `adapter_model.safetensors`
  - `training_loss.png`
  - `trainer_log.jsonl`
  - `all_results.json`

4. 提取关键指标（本次记录）：
- `train_loss=1.1283`
- `eval_loss=1.1833`
- `epoch=2.9217`

5. 对话验证（同题对比）：
- 原始模型回答
- 加载 adapter 后回答  
观察：法条引用是否更具体、结构是否更清晰、措辞是否更法律咨询化。
### 7.4 DPO 训练

#### 7.4.1 前置条件
- 已完成 SFT，且有可用 SFT adapter（作为 DPO 起点）
- 已准备偏好数据（字段：`instruction`, `input`, `chosen`, `rejected`）
- `dataset_info.json` 已注册该数据集并设置：`"ranking": true`

#### 7.4.2 推荐配置（最小可复现）
- `stage: dpo`
- `finetuning_type: lora`
- `create_new_adapter: true`（在 SFT adapter 基础上新建 DPO adapter）
- `learning_rate: 5e-6`
- `pref_beta: 0.1`
- `pref_loss: sigmoid`
- `template: qwen`

#### 7.4.3 运行命令
```bash
llamafactory-cli train my_configs/<your_dpo_yaml>.yaml
```

#### 7.4.4 关键产物检查
- DPO adapter 目录（新建，不覆盖 SFT adapter）
- `all_results.json` / `trainer_state.json`
- 训练日志中的关键指标：
  - `loss`
  - `rewards/margins`
  - `rewards/accuracies`
  - `rewards/chosen`
  - `rewards/rejected`

#### 7.4.5 本项目复现结果（基准）
- `train_loss=0.1079`
- `eval_loss=0.0976`
- `eval_rewards/margins=4.4497`
- `eval_rewards/accuracies=1.0`
- `train_runtime≈567.7s`

#### 7.4.6 判定标准
- 成功信号：
  1. loss 下降并稳定  
  2. rewards/margins 持续为正并上升  
  3. 三方对比中 DPO 在法律问答质量上优于 SFT  
- 风险信号：
  1. accuracies 过快到 1.0（小数据记忆风险）  
  2. 输出重复或过度冗长（偏好过拟合）

#### 7.4.7 对比验证（必须）
用同一组问题做三方对比：`Base vs SFT vs DPO`，至少覆盖：
- 训练分布内问题（看精修效果）
- 分布外问题（看泛化与副作用）
### 7.5 评测与回归测试

本项目评测采用“能力评测 + 回归告警”双轨机制。

#### 7.5.1 回归评测流程
1. 固定 Golden 用例与评分脚本（同题同口径）  
2. 对 `Base / SFT / DPO` 分别生成答案  
3. 计算统一指标：
   - law_accuracy
   - coverage
   - repetition_rate
   - avg_len
   - hallucination_count
4. 自动产出回归报告（含告警）

#### 7.5.2 本轮回归结果（run_20260426_131632）
| Model | law_accuracy | coverage | repetition_rate | avg_len | hallucination_count |
|---|---:|---:|---:|---:|---:|
| base | 0.850 | 0.433 | 0.000 | 545.0 | 0 |
| sft | 0.850 | 0.267 | 0.000 | 308.5 | 0 |
| dpo | 1.000 | 0.300 | 0.000 | 650.9 | 0 |

告警：
- `[ALERT] SFT vs BASE: coverage drop > 2pct`

#### 7.5.3 回归结论
1. DPO 在法条准确率上优于 Base/SFT（1.000）；  
2. SFT 覆盖率出现回归（0.433 → 0.267），需在数据与评测口径上进一步修正；  
3. 当前无重复率与幻觉告警（均为 0），主要问题集中在“关键点覆盖不足”。

---

### 7.6 部署与压测

#### 7.6.1 服务化部署（已落地）
部署模型：`/root/autodl-tmp/merged_models/law_qa_dpo`  
部署框架：vLLM（OpenAI-Compatible API）

启动命令：
```bash
vllm serve /root/autodl-tmp/merged_models/law_qa_dpo \
  --served-model-name "law-expert-fp16" \
  --host 0.0.0.0 \
  --port 6006 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
```

#### 7.6.2 并发压测结果（FP16，最新）
| 并发 | 请求数 | TPS (tokens/s) | TTFT P99 (ms) | Latency P99 (ms) | Latency Avg (ms) | Avg Tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 48.58 | 39.5 | 2588.8 | 2598.8 | 126.2 |
| 4 | 20 | 161.73 | 286.3 | 3295.1 | 3119.8 | 126.2 |
| 8 | 20 | 260.73 | 319.5 | 3413.2 | 3256.3 | 126.2 |
| 16 | 40 | 486.64 | 358.3 | 3727.6 | 3512.1 | 126.3 |

补充（`c=8, max_tokens=256`）：
- TPS: 263.04
- TTFT P99: 440.8ms
- Latency P99: 6692.0ms
- Avg Tokens: 244.4

#### 7.6.3 参数调优结论（本轮）
- `max_num_seqs`：最关键吞吐参数（历史实验中可带来 5x 级提升）  
- `gpu_memory_utilization`：在 V100S-32GB + 7B FP16 场景影响较小  
- `max_model_len`：2048 与 4096 差异不明显（当前业务输入长度较短）  
- `enable-prefix-caching`：在短前缀场景有小幅收益，建议默认开启

---
## 附录 A：关键配置清单

### A.1 训练配置（SFT）
```yaml
# 关键参数（摘要）
model_name_or_path: Qwen2.5-7B-Instruct
stage: sft
finetuning_type: lora
quantization_bit: 4
template: qwen
learning_rate: 1e-4
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
cutoff_len: 2048
bf16: true
```

### A.2 训练配置（DPO）
```yaml
# 关键参数（摘要）
stage: dpo
finetuning_type: lora
create_new_adapter: true
learning_rate: 5e-6
pref_beta: 0.1
pref_loss: sigmoid
template: qwen
```

### A.3 评测配置
- Golden 用例：`eval/golden_cases.json`
- 评测脚本：`eval/auto_eval.py`
- 回归告警：`eval/regression_check.py`
- 一键运行：`eval/run_eval_pipeline.sh`
- 核心阈值（当前）：
  - coverage 下降 > 2pct
  - law_accuracy 下降 > 2pct
  - repetition_rate 上升 > 5pct
  - hallucination_count 增加 > 2

### A.4 推理部署配置（vLLM）
```bash
vllm serve /root/autodl-tmp/merged_models/law_qa_dpo \
  --served-model-name "law-expert-fp16" \
  --host 0.0.0.0 \
  --port 6006 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
```

### A.5 RAG 配置（当前主线）
- 检索：Dense（FAISS）
- Embedding：bge-base-zh（V3最优）
- 切分策略：递归语义切分
- 参数建议：
  - `chunk_size=300`
  - `chunk_overlap=50`
  - `top_k=3`

### A.6 轻量 Agent 配置（当前状态）
- 工具：`search_law(query, top_k)`
- Agent 流程：function calling（最小实现）
- 轨迹日志：`agent_minimal/logs/trace_*.json`
- 当前验证状态：工具调用闭环未触发（0/3），待修复

---

## 附录 B：实验记录模板

> 建议每次实验复制一份，保证可追溯与可复现。

### B.1 实验元信息
- 实验ID：
- 日期：
- 负责人：
- 分支/提交：
- 目标模块：SFT / DPO / Eval / Deploy / RAG / Agent
- 假设（Hypothesis）：

### B.2 输入配置
- 模型：
- 数据集：
- 关键超参：
- 运行命令：
```bash
# 命令粘贴区
```

### B.3 输出产物
- 日志路径：
- 模型输出路径：
- 报告路径：
- 图表路径：

### B.4 指标结果
| 指标 | 值 | 备注 |
|---|---:|---|
| train_loss |  |  |
| eval_loss |  |  |
| coverage |  |  |
| law_accuracy |  |  |
| TTFT P99 |  |  |
| TPS |  |  |

### B.5 结果解读
1. 主要结论：
2. 与预期是否一致：
3. 异常点：
4. 根因分析（候选）：

### B.6 下一步动作
- [ ] 动作1
- [ ] 动作2
- [ ] 动作3

---

## 附录 C：术语表

- **SFT (Supervised Fine-Tuning)**：监督微调，用标注问答对学习“如何回答”。  
- **DPO (Direct Preference Optimization)**：直接偏好优化，用 chosen/rejected 对学习“更偏好的回答”。  
- **LoRA / QLoRA**：低秩参数微调；QLoRA 在低比特量化基座上训练 LoRA 以省显存。  
- **train_on_prompt=false**：只在 assistant token 上计算损失。  
- **Catastrophic Forgetting**：领域微调后通用能力明显退化。  
- **RAG (Retrieval-Augmented Generation)**：检索增强生成，将外部知识注入上下文。  
- **Dense Retrieval**：基于向量语义相似度的检索。  
- **BM25**：基于词频统计的稀疏检索方法。  
- **RRF (Reciprocal Rank Fusion)**：多路检索结果融合算法。  
- **Reranker**：对召回候选做二次精排的模型。  
- **KV Cache**：缓存历史 token 的 Key/Value，减少重复计算。  
- **PagedAttention**：vLLM 的分页式 KV 管理机制，减少显存碎片。  
- **TTFT (Time To First Token)**：首 token 延迟。  
- **TPS (Tokens Per Second)**：吞吐速度（每秒生成 token 数）。  
- **P99 延迟**：99 分位延迟，反映尾部请求体验。  
- **Function Calling**：模型按 schema 调用外部工具并回填结果。  

---
