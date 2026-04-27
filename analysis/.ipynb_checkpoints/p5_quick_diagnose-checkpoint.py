#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import statistics as st

RUN_DIR = "/root/autodl-tmp/saves/p5_diag_smoke"
TS_PATH = os.path.join(RUN_DIR, "trainer_state.json")

def trend(xs):
    # 简单趋势：后半均值 - 前半均值
    if len(xs) < 4:
        return 0.0
    mid = len(xs)//2
    return (sum(xs[mid:]) / max(1, len(xs[mid:]))) - (sum(xs[:mid]) / max(1, len(xs[:mid])))

def main():
    if not os.path.exists(TS_PATH):
        print(f"❌ not found: {TS_PATH}")
        return

    d = json.load(open(TS_PATH, "r", encoding="utf-8"))
    logs = d.get("log_history", [])

    train_losses = [x["loss"] for x in logs if "loss" in x and isinstance(x["loss"], (int, float))]
    eval_losses = [x["eval_loss"] for x in logs if "eval_loss" in x and isinstance(x["eval_loss"], (int, float))]

    if not train_losses:
        print("❌ no train loss found")
        return

    t_std = st.pstdev(train_losses) if len(train_losses) > 1 else 0.0
    t_trend = trend(train_losses)
    e_trend = trend(eval_losses) if eval_losses else 0.0

    best_eval = d.get("best_metric", eval_losses[-1] if eval_losses else None)
    final_train = train_losses[-1]
    final_eval = eval_losses[-1] if eval_losses else None

    print("=== P5 Quick Diagnosis ===")
    print(f"train_loss_count={len(train_losses)}")
    print(f"eval_loss_count={len(eval_losses)}")
    print(f"train_loss_std={t_std:.4f}")
    print(f"train_loss_trend={t_trend:.4f} (negative is good)")
    print(f"eval_loss_trend={e_trend:.4f} (negative is good)")
    print(f"final_train_loss={final_train:.4f}")
    print(f"final_eval_loss={final_eval if final_eval is not None else 'NA'}")
    print(f"best_eval_loss={best_eval}")

    # 规则判定（启发式）
    diagnosis = []
    if t_std > 0.20:
        diagnosis.append("震荡风险：train_loss 波动较大")
    if t_trend > 0.05:
        diagnosis.append("发散风险：train_loss 后期上升")
    if final_eval is not None and (final_eval - final_train) > 0.25 and e_trend > 0:
        diagnosis.append("过拟合风险：train 降但 eval 走差")
    if abs(t_trend) < 0.02 and final_train > 1.2:
        diagnosis.append("欠拟合风险：train_loss 降不动且水平偏高")

    if not diagnosis:
        diagnosis.append("整体稳定：未见明显震荡/发散/过拟合/欠拟合信号")

    print("\nDiagnosis:")
    for i, x in enumerate(diagnosis, 1):
        print(f"{i}. {x}")

    print("\nNext actions (fixed order): lr -> batch -> data -> rank")

if __name__ == "__main__":
    main()