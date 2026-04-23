# baseline_test.py
# 先不改任何东西，记录当前模型的真实表现

from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:6006/v1"  # 你的 vLLM 服务
)

# 用一个典型问题做基线
test_question = "我在公司工作了3年，公司突然说要裁员，我能拿到什么赔偿？"

response = client.chat.completions.create(
    model="law-expert",
    messages=[
        {"role": "system", "content": "你是一个法律专家，请根据用户的问题给出专业的回答"},  # 当前版本
        {"role": "user", "content": test_question}
    ],
    temperature=0.1,
    max_tokens=1024
)

print("=== 当前基线输出 ===")
print(response.choices[0].message.content)