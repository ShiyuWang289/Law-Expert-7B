#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from agent_fc import MinimalLawAgent

DEMO_QUESTIONS = [
    "劳动合同到期公司不续签，我能拿多少补偿？",
    "同事在公司群里造谣我贪污，我可以怎么维权？",
    "外卖骑手送餐受伤，工伤怎么认定？"
]

def main():
    os.makedirs("agent_minimal/logs", exist_ok=True)
    agent = MinimalLawAgent(
        base_url="http://127.0.0.1:6006/v1",
        model="law-expert"
    )

    for i, q in enumerate(DEMO_QUESTIONS, start=1):
        trace_file = f"agent_minimal/logs/trace_{i}.json"
        ans, trace = agent.ask(q, save_trace_path=trace_file)
        print(f"\n=== Q{i} ===")
        print("Q:", q)
        print("A:", ans[:400], "..." if len(ans) > 400 else "")
        print("trace saved:", trace_file)

if __name__ == "__main__":
    main()