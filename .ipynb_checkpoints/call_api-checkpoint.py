# call_api.py
from openai import OpenAI
import time

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:6006/v1"
)

SYSTEM_PROMPT = "你是一个专业的中国法律顾问，请根据用户的法律问题，给出准确、专业、有法律依据的回答。"

# ========================================
# 测试1：基础对话
# ========================================
print("="*60)
print("测试1：基础对话")
print("="*60)

response = client.chat.completions.create(
    model="law-expert",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "劳动合同到期公司不续签，员工能获得赔偿吗？"}
    ],
    max_tokens=512,
    temperature=0.1
)

print(f"\n回答:\n{response.choices[0].message.content}\n")
print(f"Token 用量: prompt={response.usage.prompt_tokens}, "
      f"completion={response.usage.completion_tokens}, "
      f"total={response.usage.total_tokens}")

# ========================================
# 测试2：流式输出
# ========================================
print("\n" + "="*60)
print("测试2：流式输出")
print("="*60)

stream = client.chat.completions.create(
    model="law-expert",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "竞业协议不给补偿金，我还需要遵守吗？"}
    ],
    max_tokens=256,
    temperature=0.1,
    stream=True
)

print("\n回答（流式）：", end="", flush=True)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")

# ========================================
# 测试3：批量推理速度测试
# ========================================
print("="*60)
print("测试3：推理速度测试（10次请求）")
print("="*60)

questions = [
    "什么是劳动仲裁？",
    "合同违约怎么赔偿？",
    "网购可以七天无理由退货吗？",
    "交通事故如何定责？",
    "工伤赔偿标准是什么？",
    "租房押金房东不退怎么办？",
    "离婚财产如何分割？",
    "借钱不还可以起诉吗？",
    "被诈骗如何报警？",
    "公司不缴社保怎么办？"
]

t_start = time.time()
total_tokens = 0
responses_data = []

for i, q in enumerate(questions):
    resp = client.chat.completions.create(
        model="law-expert",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q}
        ],
        max_tokens=128,
        temperature=0.1
    )
    total_tokens += resp.usage.completion_tokens
    responses_data.append({
        "question": q,
        "answer": resp.choices[0].message.content,
        "tokens": resp.usage.completion_tokens
    })
    print(f"  [{i+1}/10] {q[:20]}... → {resp.usage.completion_tokens} tokens")

t_end = time.time()
elapsed = t_end - t_start

print(f"\n{'='*60}")
print("📊 批量推理统计:")
print(f"{'='*60}")
print(f"  总耗时:       {elapsed:.1f}s")
print(f"  总 tokens:    {total_tokens}")
print(f"  平均速度:     {total_tokens/elapsed:.1f} tokens/s")
print(f"  平均每次:     {elapsed/len(questions):.2f}s/请求")
print(f"  平均每问回答: {total_tokens/len(questions):.1f} tokens")

# ========================================
# 测试4：质量抽查
# ========================================
print(f"\n{'='*60}")
print("测试4：回答质量抽查（3个示例）")
print(f"{'='*60}")

for i, data in enumerate(responses_data[:3]):
    print(f"\n【问题 {i+1}】{data['question']}")
    print(f"【回答】{data['answer'][:200]}{'...' if len(data['answer'])>200 else ''}")
    print(f"【Token数】{data['tokens']}")