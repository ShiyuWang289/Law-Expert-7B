#!/usr/bin/env python3
import argparse, time, threading, statistics, json, os
from openai import OpenAI

PROMPTS = [
    "劳动合同到期公司不续签，补偿标准是什么？",
    "被违法辞退如何主张2N赔偿？",
    "网购买到假货怎么维权？",
    "交通事故对方全责拒赔怎么办？"
]

def request_once(client, model, prompt, max_tokens=128):
    t0 = time.time()
    first = None
    out_tokens = 0
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"你是专业中国法律顾问"},
            {"role":"user","content":prompt}
        ],
        temperature=0,
        max_tokens=max_tokens,
        stream=True
    )
    for chunk in stream:
        txt = chunk.choices[0].delta.content if chunk.choices else None
        if txt:
            if first is None:
                first = time.time() - t0
            out_tokens += 1
    total = time.time() - t0
    return (first or total), total, out_tokens

def run(base_url, model, concurrency, nreq, max_tokens):
    client = OpenAI(api_key="EMPTY", base_url=base_url)
    lock = threading.Lock()
    idx = 0
    ttfts, lats, toks = [], [], []

    def worker():
        nonlocal idx
        while True:
            with lock:
                if idx >= nreq:
                    return
                i = idx
                idx += 1
            p = PROMPTS[i % len(PROMPTS)]
            ttft, lat, tok = request_once(client, model, p, max_tokens)
            with lock:
                ttfts.append(ttft); lats.append(lat); toks.append(tok)

    t0 = time.time()
    ts = [threading.Thread(target=worker) for _ in range(concurrency)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall = time.time() - t0

    def p99(xs):
        ys = sorted(xs)
        return ys[max(0, int(len(ys)*0.99)-1)] if ys else 0

    return {
        "concurrency": concurrency,
        "requests": nreq,
        "max_tokens": max_tokens,
        "tps": round(sum(toks)/wall, 2),
        "ttft_p99_ms": round(p99(ttfts)*1000, 1),
        "lat_p99_ms": round(p99(lats)*1000, 1),
        "lat_avg_ms": round(statistics.mean(lats)*1000, 1),
        "avg_tokens": round(statistics.mean(toks), 1)
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="http://127.0.0.1:6006/v1")
    ap.add_argument("--model", default="law-expert")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--tag", default="manual")
    args = ap.parse_args()

    result = run(args.base_url, args.model, args.concurrency, args.requests, args.max_tokens)
    print(result)

    os.makedirs("bench/results", exist_ok=True)
    out = f"bench/results/{args.tag}_c{args.concurrency}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("saved:", out)