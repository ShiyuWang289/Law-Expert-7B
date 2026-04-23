# performance_summary.py
"""
请根据 call_api.py 运行的实际数据，手动填入以下变量。
"""

# 从 call_api.py 的输出中填入实际数据
# 例如：vllm_tps_actual = 65.4
vllm_tps_actual = 48.5 # <--- 在这里填入 call_api.py 输出的 "平均速度" (tokens/s)
avg_response_time_actual = 2.64 # <--- 在这里填入 call_api.py 输出的 "平均响应时间" (s/请求)
avg_tokens_per_response_actual = 128.0 # <--- 在这里填入 call_api.py 输出的 "平均每问回答" (tokens)


# 假设的 bf16 (transformers 原生) 基线速度。
# 如果你之前没有实际测试过 transformers 框架的推理速度，可以使用这个经验值。
# 实际速度可能更高或更低，取决于模型和硬件。
bf16_tps_baseline = 20.0 # 假设 transformers 原生推理是 20 tokens/s

if vllm_tps_actual is not None:
    speedup = vllm_tps_actual / bf16_tps_baseline
else:
    speedup = None # 如果没有 vllm_tps_actual，则无法计算加速比

print("\n" + "="*65)
print("📊 vLLM 部署性能总结（FP16 @ V100S）")
print("="*65)
print(f"{'指标':<25} {'数值':>20}")
print("-"*65)
print(f"  {'模型精度':<23} {'FP16':>20}")
print(f"  {'GPU 型号':<23} {'Tesla V100S-32GB':>20}")
print(f"  {'显存占用':<23} {'~15GB':>20}") # V100S 32GB 显存足以容纳 15GB 的 FP16 模型

print(f"  {'推理速度 (vLLM)':<23} {f'{vllm_tps_actual:.1f} tokens/s' if vllm_tps_actual else '待填入':>20}")
print(f"  {'基线速度 (Transformers)':<23} {f'{bf16_tps_baseline:.1f} tokens/s (估计值)':>20}")
print(f"  {'加速比 (vLLM vs Transformers)':<23} {f'{speedup:.1f}x' if speedup else '待填入':>20}")

print(f"  {'平均响应时间':<23} {f'{avg_response_time_actual:.2f}s/请求' if avg_response_time_actual else '待填入':>20}")
print(f"  {'平均回答长度':<23} {f'{avg_tokens_per_response_actual:.0f} tokens' if avg_tokens_per_response_actual else '待填入':>20}")
print(f"  {'并发支持':<23} {'✅ (vLLM Batching)':>20}")
print(f"  {'流式输出':<23} {'✅':>20}")
print(f"  {'API 兼容性':<23} {'OpenAI Compatible':>20}")
print("="*65)

