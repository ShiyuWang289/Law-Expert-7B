#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob, json

files = sorted(glob.glob("agent_minimal/logs/trace_*.json"))
if not files:
    print("No trace files found.")
    raise SystemExit(1)

ok_total = 0
print("| trace | has_tool_call | has_tool_result | has_final_answer |")
print("|---|---|---|---|")
for f in files:
    x = json.load(open(f, "r", encoding="utf-8"))
    stages = [s.get("stage") for s in x.get("steps", [])]
    has_tool_call = "llm_decide" in stages and any(s.get("message", {}).get("tool_calls") for s in x.get("steps", []) if s.get("stage")=="llm_decide")
    has_tool_result = "tool_exec" in stages
    has_final = "final_answer" in stages or "final_answer_no_tool" in stages
    ok = has_tool_result and has_final
    ok_total += int(ok)
    print(f"| {f} | {has_tool_call} | {has_tool_result} | {has_final} |")

print(f"\nPass: {ok_total}/{len(files)} traces have tool execution + final answer.")