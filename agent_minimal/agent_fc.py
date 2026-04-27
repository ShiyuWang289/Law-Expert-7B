#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from openai import OpenAI
from search_tool import LawSearcher

SYSTEM_PROMPT = """你是法律问答助手。
当问题需要法律依据时，优先调用 search_law 工具检索法条后再回答。
回答要包含：
1) 结论
2) 法律依据
3) 维权/操作建议
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_law",
            "description": "检索最相关的法律条文片段",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "用户法律问题"},
                    "top_k": {"type": "integer", "description": "返回条数，建议1-5", "default": 3}
                },
                "required": ["query"]
            }
        }
    }
]

class MinimalLawAgent:
    def __init__(self, base_url="http://127.0.0.1:6006/v1", model="law-expert-fp16"):
        self.client = OpenAI(api_key="EMPTY", base_url=base_url)
        self.model = model
        self.searcher = LawSearcher()

    def _tool_call(self, name, args):
        if name != "search_law":
            return {"error": f"unknown tool: {name}"}
        query = args.get("query", "")
        top_k = int(args.get("top_k", 3))
        return self.searcher.search_law(query, top_k=top_k)

    def ask(self, user_query: str, save_trace_path=None):
        trace = {"user_query": user_query, "steps": []}
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]

        # Step1: 让模型决定是否调用工具
        resp1 = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0
        )
        msg = resp1.choices[0].message
        trace["steps"].append({"stage": "llm_decide", "message": msg.model_dump()})

        # Step2: 若有工具调用 -> 执行 -> 注入
        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                tool_result = self._tool_call(fn, args)
                trace["steps"].append({
                    "stage": "tool_exec",
                    "tool_name": fn,
                    "tool_args": args,
                    "tool_result": tool_result
                })

                messages.append(msg.model_dump())
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn,
                    "content": json.dumps(tool_result, ensure_ascii=False)
                })

            # Step3: 二次生成最终答案
            resp2 = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0
            )
            final_answer = resp2.choices[0].message.content
            trace["steps"].append({"stage": "final_answer", "answer": final_answer})
        else:
            final_answer = msg.content
            trace["steps"].append({"stage": "final_answer_no_tool", "answer": final_answer})

        if save_trace_path:
            with open(save_trace_path, "w", encoding="utf-8") as f:
                json.dump(trace, f, ensure_ascii=False, indent=2)

        return final_answer, trace