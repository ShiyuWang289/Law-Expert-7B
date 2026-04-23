# locustfile.py（修复版）
import random
import time
import json
from locust import HttpUser, task, between

LAW_QUESTIONS = [
    "劳动合同到期公司不续签，员工能获得赔偿吗？",
    "被公司违法辞退，能要求双倍赔偿吗？",
    "公司拖欠工资两个月，我能去劳动局投诉吗？",
    "签了竞业协议但公司不给补偿金，还需要遵守吗？",
    "工伤认定的条件是什么？",
    "网购商品与描述不符可以退货吗？",
    "朋友借钱不还，超过三年还能起诉吗？",
    "交通事故责任认定不服怎么申请复核？",
    "租房合同未到期房东要求搬走合法吗？",
    "离婚后孩子的抚养权如何判定？",
    "公司未足额缴纳社保怎么维权？",
    "遭遇电信诈骗应该怎么处理？",
]
SYSTEM_PROMPT = "你是一个专业的中国法律顾问，请给出准确、专业的回答。"

class LawExpertUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def ask_law_question(self):
        question = random.choice(LAW_QUESTIONS)
        payload = {
            "model": "law-expert",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question}
            ],
            "max_tokens": 256,
            "temperature": 0.1,
            "stream": False     # 保持 False，通过加超时解决
        }

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            timeout=120,        # ★ 修复1：设置 120 秒超时
            catch_response=True,
            name="/v1/chat/completions [law_qa]"   # ★ 修复2：去掉 POST 前缀
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    if len(content) > 5:
                        response.success()
                    else:
                        response.failure("回答内容过短")
                except Exception as e:
                    response.failure(f"解析失败: {e}")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def ask_short_question(self):
        question = random.choice([
            "什么是劳动仲裁？",
            "竞业协议多久失效？",
            "工伤怎么申请？"
        ])
        payload = {
            "model": "law-expert",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question}
            ],
            "max_tokens": 128,
            "temperature": 0.1,
        }

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            timeout=120,        # ★ 修复1：设置 120 秒超时
            catch_response=True,
            name="/v1/chat/completions [short_qa]"  # ★ 修复2：去掉 POST 前缀
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")