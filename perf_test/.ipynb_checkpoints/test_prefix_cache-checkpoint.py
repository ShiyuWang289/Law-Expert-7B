# test_prefix_cache.py
import asyncio, time, aiohttp, json

URL = "http://localhost:6006/v1/chat/completions"

# 长 System Prompt（越长缓存效果越明显）
LONG_SYSTEM = """你是一个专业的中国法律顾问，拥有20年执业经验，熟悉劳动法、合同法、
民法典、刑法、行政法等各领域法律。请根据用户的法律问题，给出准确、专业、
有具体法律条文依据的回答。回答时需要：1）引用具体法律条文；2）给出明确结论；
3）提供可操作的维权建议；4）说明注意事项和时间限制。"""

QUESTIONS = [
    "劳动合同到期公司不续签，员工能获得赔偿吗？",
    "被公司违法辞退，能要求双倍赔偿吗？",
    "公司拖欠工资两个月，我能去劳动局投诉吗？",
    "签了竞业协议但公司不给补偿金，还需要遵守吗？",
    "工伤认定的条件是什么？",
]

async def measure_ttft(session, question, system):
    payload = {
        "model": "law-expert",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": question}
        ],
        "max_tokens": 128, "temperature": 0.1, "stream": True
    }
    t_start = time.time()
    first_token_time = None
    
    async with session.post(URL, json=payload) as resp:
        async for line in resp.content:
            line = line.decode().strip()
            if line.startswith("data:") and line != "data: [DONE]":
                try:
                    data = json.loads(line[5:])
                    if data["choices"][0]["delta"].get("content"):
                        if first_token_time is None:
                            first_token_time = time.time()
                            break
                except:
                    pass
    
    return (first_token_time - t_start) * 1000 if first_token_time else -1

async def main():
    async with aiohttp.ClientSession() as session:
        print("="*55)
        print("测试前缀缓存效果（长 System Prompt）")
        print("="*55)
        
        print("\n🔥 预热请求（建立缓存）...")
        await measure_ttft(session, QUESTIONS[0], LONG_SYSTEM)
        
        print("\n📊 第一轮（首次/冷启动）:")
        cold_ttfts = []
        for q in QUESTIONS:
            # 用短 system 模拟无缓存命中
            ttft = await measure_ttft(session, q, "你是法律助手。" + q[:5])
            cold_ttfts.append(ttft)
            print(f"  TTFT = {ttft:.1f}ms | {q[:20]}...")
        
        print("\n📊 第二轮（相同 System Prompt，命中缓存）:")
        warm_ttfts = []
        for q in QUESTIONS:
            ttft = await measure_ttft(session, q, LONG_SYSTEM)
            warm_ttfts.append(ttft)
            print(f"  TTFT = {ttft:.1f}ms | {q[:20]}...")
        
        avg_cold = sum(cold_ttfts) / len(cold_ttfts)
        avg_warm = sum(warm_ttfts) / len(warm_ttfts)
        print(f"\n  冷启动 TTFT 均值: {avg_cold:.1f}ms")
        print(f"  缓存命中 TTFT 均值: {avg_warm:.1f}ms")
        print(f"  TTFT 降低: {(1 - avg_warm/avg_cold)*100:.1f}%  ← 前缀缓存效果")

asyncio.run(main())