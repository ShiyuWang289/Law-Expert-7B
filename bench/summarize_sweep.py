#!/usr/bin/env python3
import glob, json, argparse, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(f"bench/results/{args.tag}_c*.json"))
    rows = []
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    rows = sorted(rows, key=lambda x: x["concurrency"])

    print("| 并发 | 请求数 | TPS | TTFT P99(ms) | Latency P99(ms) | Latency Avg(ms) | Avg Tokens |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['concurrency']} | {r['requests']} | {r['tps']} | {r['ttft_p99_ms']} | {r['lat_p99_ms']} | {r['lat_avg_ms']} | {r['avg_tokens']} |")

    os.makedirs("bench/reports", exist_ok=True)
    md = f"bench/reports/{args.tag}_summary.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write(f"# Sweep Summary: {args.tag}\n\n")
        f.write("| 并发 | 请求数 | TPS | TTFT P99(ms) | Latency P99(ms) | Latency Avg(ms) | Avg Tokens |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['concurrency']} | {r['requests']} | {r['tps']} | {r['ttft_p99_ms']} | {r['lat_p99_ms']} | {r['lat_avg_ms']} | {r['avg_tokens']} |\n")
    print("saved:", md)

if __name__ == "__main__":
    main()