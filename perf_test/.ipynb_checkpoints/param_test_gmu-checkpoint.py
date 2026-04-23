# param_test_gmu.py
import asyncio, json
from metrics_client import run_concurrent_benchmark

URL = "http://localhost:6006/v1/chat/completions"
SYSTEM = "你是一个专业的中国法律顾问。"

payloads = [
    {"model": "law-expert",
     "messages": [{"role": "system", "content": SYSTEM},
                  {"role": "user",   "content": "劳动合同到期公司不续签，员工能获得赔偿吗？"}],
     "max_tokens": 256, "temperature": 0.1}
] * 20

async def main():
    report = await run_concurrent_benchmark(URL, payloads, concurrency=8, label="当前配置_c8")
    result = report.summary("当前配置_c8")
    
    # 追加保存（手动修改 label 标注当前参数）
    label = input("\n输入本次配置标签（如 gmu_0.85）：").strip()
    try:
        with open("results/param_tests.json") as f:
            all_data = json.load(f)
    except FileNotFoundError:
        all_data = {}
    all_data[label] = result
    with open("results/param_tests.json", "w") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 已保存为 '{label}'")

asyncio.run(main())