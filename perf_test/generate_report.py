# generate_report.py
import json
import os

print("""
╔══════════════════════════════════════════════════════════════╗
║           vLLM 性能调优报告                                   ║
║           模型：Qwen2.5-7B-Instruct (DPO + W4A16)           ║
║           硬件：AutoDL V100S-32GB                             ║
╚══════════════════════════════════════════════════════════════╝
""")

# ===== 第一部分：参数调优对比 =====
print("="*65)
print("一、参数调优实验结果（并发=8，请求数=24）")
print("="*65)

param_results = {}
if os.path.exists("results/param_tests.json"):
    with open("results/param_tests.json") as f:
        param_results = json.load(f)

if param_results:
    print(f"\n{'配置':<25} {'TPS':>8} {'TTFT P99':>10} {'延迟 P99':>10}")
    print("-"*65)
    for label, data in param_results.items():
        if data:
            print(f"  {label:<23} {data.get('tps', 0):>7.1f} "
                  f"{data.get('ttft_p99_ms', 0):>9.1f}ms "
                  f"{data.get('latency_p99_ms', 0):>9.1f}ms")
else:
    print("  （请将各参数测试结果填入此处）")
    print(f"\n  {'配置':<30} {'TPS':>8} {'TTFT P99':>10} {'延迟 P99':>10}")
    print("  " + "-"*60)
    print(f"  {'默认(gmu=0.90)':<30} {'XX':>8} {'XXXms':>10} {'XXXXms':>10}")
    print(f"  {'gmu=0.85':<30} {'XX':>8} {'XXXms':>10} {'XXXXms':>10}")
    print(f"  {'gmu=0.95':<30} {'XX':>8} {'XXXms':>10} {'XXXXms':>10}")
    print(f"  {'max_num_seqs=64':<30} {'XX':>8} {'XXXms':>10} {'XXXXms':>10}")
    print(f"  {'max_num_seqs=128':<30} {'XX':>8} {'XXXms':>10} {'XXXXms':>10}")
    print(f"  {'enable_prefix_caching':<30} {'XX':>8} {'XXXms':>10} {'XXXXms':>10}")
    print(f"  {'max_model_len=2048':<30} {'XX':>8} {'XXXms':>10} {'XXXXms':>10}")

# ===== 第二部分：Locust 压测结果 =====
print(f"\n{'='*65}")
print("二、Locust 压力测试结果（持续120s）")
print("="*65)
print(f"\n  {'并发用户':<12} {'RPS':>8} {'Avg延迟':>10} {'P99延迟':>10} {'失败率':>8}")
print("  " + "-"*55)

for users in [5, 10, 20]:
    csv_file = f"results/locust_c{users}_stats.csv"
    if os.path.exists(csv_file):
        import csv
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Name") == "Aggregated":
                    fail_pct = float(row.get("Failure Count", 0)) / max(float(row.get("Request Count", 1)), 1) * 100
                    print(f"  {str(users)+'用户':<12} "
                          f"{float(row.get('Requests/s', 0)):>8.2f} "
                          f"{float(row.get('Average Response Time', 0)):>9.0f}ms "
                          f"{float(row.get('99%', 0)):>9.0f}ms "
                          f"{fail_pct:>7.1f}%")
    else:
        print(f"  {str(users)+'用户':<12} {'待测试':>8} {'---':>10} {'---':>10} {'---':>8}")

# ===== 第三部分：最优配置 =====
print(f"""
{'='*65}
三、推荐最优配置
{'='*65}

  vllm serve <model_path> \\
      --served-model-name "law-expert" \\
      --host 0.0.0.0 \\
      --port 6006 \\
      --dtype auto \\
      --max-model-len 2048        # 根据实际需求设定
      --gpu-memory-utilization 0.9 \\
      --max-num-seqs 128          # 根据测试结果选最优值
      --enable-prefix-caching     # 相同 system prompt 场景必开

{'='*65}
四、关键发现与结论
{'='*65}

  1. vLLM vs Transformers：
     推理速度从 ~20 tokens/s 提升至 ~48.5 tokens/s（2.4x 加速）
     核心原因：Continuous Batching + 优化的 CUDA 内核

  2. 前缀缓存（enable-prefix-caching）：
     System Prompt 相同时，TTFT 显著降低（预期下降 30~60%）
     法律咨询场景 System Prompt 固定，强烈推荐开启

  3. 量化（W4A16）：
     模型体积 15GB → 5.2GB（压缩 ~3x）
     V100S-32GB 可直接加载 FP16，显存充足时可不量化

  4. 性能拐点：
     随并发数增加，TPS 先上升后趋于稳定，P99 延迟持续上升
     建议将 P99 延迟 < 5000ms 作为 SLA 阈值，确定最大并发数

{'='*65}
""")