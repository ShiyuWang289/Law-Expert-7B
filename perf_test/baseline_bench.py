# baseline_bench.py
import asyncio
import json
import sys
sys.path.insert(0, '.')
from metrics_client import run_concurrent_benchmark

URL = "http://localhost:6006/v1/chat/completions"
SYSTEM = "你是一个专业的中国法律顾问，请给出准确、专业的回答。"

# 测试问题（覆盖短/中/长三种输出长度）
QUESTIONS = {
    "short":  "竞业协议多久失效？",
    "medium": "劳动合同到期公司不续签，员工能获得赔偿吗？",
    "long":   "详细介绍工伤认定的完整流程，包括申请材料、时限要求和注意事项。",
}

def make_payload(question, max_tokens=256):
    return {
        "model": "law-expert",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": question}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

async def main():
    all_reports = {}
    
    # 测试1：串行（concurrency=1），测量最优单请求延迟
    payloads = [make_payload(QUESTIONS["medium"])] * 10
    report = await run_concurrent_benchmark(URL, payloads, concurrency=1, label="基线_串行(c=1)")
    all_reports["baseline_c1"] = report.summary("基线_串行(c=1)")
    
    # 测试2：低并发（concurrency=4）
    payloads = [make_payload(q) for q in list(QUESTIONS.values()) * 4]
    report = await run_concurrent_benchmark(URL, payloads, concurrency=4, label="基线_低并发(c=4)")
    all_reports["baseline_c4"] = report.summary("基线_低并发(c=4)")
    
    # 测试3：中并发（concurrency=8）
    payloads = [make_payload(q) for q in list(QUESTIONS.values()) * 8]
    report = await run_concurrent_benchmark(URL, payloads, concurrency=8, label="基线_中并发(c=8)")
    all_reports["baseline_c8"] = report.summary("基线_中并发(c=8)")
    
    # 保存基线结果
    with open("results/baseline.json", "w") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)
    print("\n✅ 基线数据已保存 → results/baseline.json")

asyncio.run(main())