from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:6006/v1")

resp = client.chat.completions.create(
    model="law-expert",
    messages=[
        {"role":"system","content":"你是专业中国法律顾问"},
        {"role":"user","content":"公司违法辞退，如何维权并主张赔偿？"}
    ],
    temperature=0,
    max_tokens=128
)

print(resp.choices[0].message.content)