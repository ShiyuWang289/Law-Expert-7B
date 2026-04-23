# metrics_client.py
# 可复用的指标采集工具，后续所有测试用它

import time
import asyncio
import aiohttp
import json
import statistics
from dataclasses import dataclass, field
from typing import List

@dataclass
class RequestResult:
    success: bool
    ttft_ms: float = 0          # 首字延迟（毫秒）
    total_time_ms: float = 0    # 总耗时（毫秒）
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""

@dataclass
class BenchmarkReport:
    results: List[RequestResult] = field(default_factory=list)
    
    def summary(self, label=""):
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        if not successful:
            print(f"❌ {label}：所有请求均失败！")
            return {}
        
        ttfts       = [r.ttft_ms for r in successful]
        totals      = [r.total_time_ms for r in successful]
        out_tokens  = [r.output_tokens for r in successful]
        total_time  = max(r.total_time_ms for r in self.results) / 1000
        
        tps = sum(out_tokens) / total_time if total_time > 0 else 0
        
        report = {
            "label":         label,
            "total_reqs":    len(self.results),
            "success_reqs":  len(successful),
            "fail_reqs":     len(failed),
            "tps":           round(tps, 2),
            "ttft_avg_ms":   round(statistics.mean(ttfts), 1),
            "ttft_p50_ms":   round(statistics.median(ttfts), 1),
            "ttft_p99_ms":   round(sorted(ttfts)[int(len(ttfts)*0.99)], 1),
            "latency_avg_ms":round(statistics.mean(totals), 1),
            "latency_p50_ms":round(statistics.median(totals), 1),
            "latency_p99_ms":round(sorted(totals)[int(len(totals)*0.99)], 1),
            "avg_out_tokens":round(statistics.mean(out_tokens), 1),
        }
        
        print(f"\n{'='*55}")
        print(f"📊 {label}")
        print(f"{'='*55}")
        print(f"  请求总数:      {report['total_reqs']}（失败 {report['fail_reqs']}）")
        print(f"  ─────────────────────────────────────────")
        print(f"  吞吐量(TPS):   {report['tps']:.1f} tokens/s")
        print(f"  ─────────────────────────────────────────")
        print(f"  TTFT 均值:     {report['ttft_avg_ms']:.1f} ms")
        print(f"  TTFT P50:      {report['ttft_p50_ms']:.1f} ms")
        print(f"  TTFT P99:      {report['ttft_p99_ms']:.1f} ms  ← 关键指标")
        print(f"  ─────────────────────────────────────────")
        print(f"  总延迟 均值:   {report['latency_avg_ms']:.1f} ms")
        print(f"  总延迟 P50:    {report['latency_p50_ms']:.1f} ms")
        print(f"  总延迟 P99:    {report['latency_p99_ms']:.1f} ms  ← 关键指标")
        print(f"{'='*55}")
        
        return report


async def single_request(session, url, payload, timeout=60):
    """发送单个流式请求，精确测量 TTFT"""
    result = RequestResult(success=False)
    t_start = time.time()
    first_token_time = None
    output_text = ""
    
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                result.error = f"HTTP {resp.status}"
                return result
            
            async for line in resp.content:
                line = line.decode().strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data["choices"][0]["delta"].get("content", "")
                    if delta:
                        if first_token_time is None:
                            first_token_time = time.time()
                        output_text += delta
                except:
                    continue
        
        t_end = time.time()
        result.success = True
        result.ttft_ms = (first_token_time - t_start) * 1000 if first_token_time else 0
        result.total_time_ms = (t_end - t_start) * 1000
        result.output_tokens = len(output_text.split())  # 近似值
        
    except Exception as e:
        result.error = str(e)
    
    return result


async def run_concurrent_benchmark(
    url, payloads, concurrency, label="benchmark"
):
    """并发压测核心函数"""
    report = BenchmarkReport()
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded_request(payload):
        async with semaphore:
            return await single_request(
                session, url, {**payload, "stream": True}
            )
    
    print(f"\n🚀 开始测试: {label}（并发={concurrency}，请求数={len(payloads)}）")
    
    async with aiohttp.ClientSession() as session:
        tasks = [bounded_request(p) for p in payloads]
        results = await asyncio.gather(*tasks)
    
    report.results = list(results)
    return report