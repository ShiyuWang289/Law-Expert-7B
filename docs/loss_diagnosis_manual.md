# P5 Loss 诊断手册（现象 → 排查动作）

> 目标：把“调参”变成“诊断系统”。  
> 适用：SFT/QLoRA（LLaMA-Factory），单卡 V100S 快速定位。

---

## 1. 诊断总流程（先判现象，再做单变量排查）

1) 先看曲线与指标  
- 读取 `train_loss`、`eval_loss`、`grad_norm`（若有）  
- 判断属于：震荡 / 发散 / 过拟合 / 欠拟合

2) 固定数据与随机种子  
- 避免把数据波动误判为超参问题

3) 单变量排查顺序（强制）  
- **lr → batch(含grad_accum) → data → rank**  
- 每次只改 1 个变量，跑短实验（≤10分钟）

4) 记录结论  
- 每一步保留“改动-结果-结论”，形成可复盘证据链

---

## 2. 现象 → 排查动作（执行表）

| 现象 | 典型信号 | 第一优先排查 | 第二优先排查 | 第三优先排查 | 第四优先排查 | 建议动作 |
|---|---|---|---|---|---|---|
| 震荡（oscillation） | train_loss 上下大幅波动，eval_loss 不稳定 | lr 过大 | 有效 batch 太小 | 数据噪声/混杂风格 | LoRA rank 过高导致不稳 | lr 下调 2x；增大 grad_accum；清理异常样本；rank 16→8 |
| 发散（divergence） | loss 持续上升/出现 NaN/Inf | lr 明显过大 | 数值稳定（bf16/fp16、grad clip） | 脏数据（超长/异常 token） | rank 与学习率组合过激 | lr 下调到 1/2~1/4；开 `max_grad_norm=1.0`；检查异常样本 |
| 过拟合（overfit） | train_loss 继续降，eval_loss 回升 | 数据量不足/分布偏 | 正则不足（dropout） | 训练步数过多 | rank 过高记忆化 | 提前停止；增 val；lora_dropout↑；rank 16→8；补数据 |
| 欠拟合（underfit） | train_loss 和 eval_loss 都高且降不动 | lr 太小 | 训练步数不足 | rank 太低容量不够 | 数据质量不足 | lr 上调 1.5~2x；epoch↑；rank 8→16；提升样本质量 |

---

## 3. 四大变量的“排除法模板”

### 3.1 LR 排查（第一优先）
- 快速网格：`5e-5 / 1e-4 / 2e-4`
- 判据：最低 `eval_loss` + 最低波动度（train loss std）
- 结论模板：  
  - 若大 lr 更好且不抖：可保留大 lr  
  - 若大 lr 抖动明显：回退到中档 lr

### 3.2 Batch 排查（第二优先）
- 固定“有效 batch”与“单卡 batch”分别测试：
  - `per_device_train_batch_size=2, grad_accum=2`
  - `per_device_train_batch_size=1, grad_accum=4`
- 判据：同等吞吐下，哪组波动更小、eval_loss更低

### 3.3 Data 排查（第三优先）
- 抽查高 loss 样本（长文本、噪声、模板混乱）
- 做小规模“净化集”对照（如 120 样本）
- 判据：清洗后若曲线显著平稳，优先归因数据问题

### 3.4 Rank 排查（第四优先）
- 测试 `rank=8/16`
- 判据：rank↑若只降 train_loss 不降 eval_loss，多为过拟合倾向

---

## 4. 快速诊断实验规范（V100S，≤10分钟）

- `max_samples: 120`
- `cutoff_len: 512`
- `num_train_epochs: 0.3`
- `eval_steps: 20`
- `save_steps: 9999`（减少I/O）
- 输出：`train_loss`、`eval_loss`、runtime、波动度

---

## 5. 你项目当前可复述结论（可写面试）

1. 已建立“现象→动作→证据”的损失诊断闭环；  
2. 排查顺序固定为 `lr → batch → data → rank`，避免同时改多个参数导致归因失败；  
3. 通过小样本短跑（10分钟级）先定方向，再做完整训练验证，提升实验效率与可靠性。