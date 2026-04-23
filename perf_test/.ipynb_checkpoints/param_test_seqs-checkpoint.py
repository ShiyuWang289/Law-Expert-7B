# param_test_seqs.py
import asyncio, json
from metrics_client import run_concurrent_benchmark

URL = "http://localhost:6006/v1/chat/completions"
SYSTEM = "你是一个专业的中国法律顾问。"

def make_payloads(n):
    questions = [
        "什么是劳动仲裁？", "合同违约怎么赔偿？", "网购可以七天退货吗？",
        "工伤怎么认定？", "借钱不还如何起诉？"
    ]
    return [
        {"model": "law-expert",
         "messages": [{"role": "system", "content": SYSTEM},
                      {"role": "user",   "content": questions[i % len(questions)]}],
         "max_tokens": 128, "temperature": 0.1}
        for i in range(n)
    ]

async def main():
    # 测试不同并发压力
    for concurrency in [4, 8, 16]:
        payloads = make_payloads(concurrency * 3)
        report = await run_concurrent_benchmark(
            URL, payloads, concurrency=concurrency,
            label=f"当前max_num_seqs_c{concurrency}"
        )
        report.summary(f"并发={concurrency}")
        await asyncio.sleep(2)  # 冷却间隔

asyncio.run(main())