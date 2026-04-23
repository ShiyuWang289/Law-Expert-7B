# domain_eval.py
# 使用方法：python domain_eval.py --model [base|sft|dpo]
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os

# ========================================
# 10 个法律领域 case（核心评估内容）
# ========================================
TEST_CASES = [
    # ① 劳动法类（训练集内）
    {
        "id": "labor_01",
        "category": "劳动法",
        "type": "训练集内",
        "question": "劳动合同到期公司不续签，员工能获得赔偿吗？赔偿标准是多少？",
        "key_points": ["《劳动合同法》第四十六条", "每满一年一个月工资", "六个月以上按一年计算"]
    },
    {
        "id": "labor_02",
        "category": "劳动法",
        "type": "训练集内",
        "question": "被公司违法辞退，能要求双倍赔偿吗？",
        "key_points": ["《劳动合同法》第八十七条", "2N赔偿", "违法解除"]
    },
    # ② 劳动法类（训练集外）
    {
        "id": "labor_03",
        "category": "劳动法",
        "type": "训练集外",
        "question": "公司拖欠工资两个月，我能直接去劳动局投诉吗？",
        "key_points": ["劳动监察", "劳动仲裁", "追讨时效"]
    },
    # ③ 合同法类
    {
        "id": "contract_01",
        "category": "合同法",
        "type": "训练集外",
        "question": "口头协议有法律效力吗？发生纠纷怎么举证？",
        "key_points": ["合同形式", "口头合同有效", "举证责任"]
    },
    {
        "id": "contract_02",
        "category": "合同法",
        "type": "训练集内",
        "question": "租房合同未到期，房东突然涨价合理吗？我有什么权利？",
        "key_points": ["合同约束力", "违约责任", "继续履行"]
    },
    # ④ 民法类
    {
        "id": "civil_01",
        "category": "民法",
        "type": "训练集外",
        "question": "朋友向我借款一万元，没有借条，三年后拒不还款，还能起诉吗？",
        "key_points": ["诉讼时效三年", "证据", "小额诉讼"]
    },
    {
        "id": "civil_02",
        "category": "民法",
        "type": "训练集内",
        "question": "遭遇电信诈骗损失五千元，能追回来吗？应该怎么做？",
        "key_points": ["立即报警", "冻结账户", "证据收集"]
    },
    # ⑤ 交通/侵权类
    {
        "id": "traffic_01",
        "category": "交通法",
        "type": "训练集内",
        "question": "发生交通事故后，对方全责但拒绝赔偿怎么办？",
        "key_points": ["责任认定书", "保险理赔", "民事诉讼"]
    },
    # ⑥ 消费者权益
    {
        "id": "consumer_01",
        "category": "消费者权益",
        "type": "训练集内",
        "question": "网购买到假冒商品，平台和商家都不处理，怎么维权？",
        "key_points": ["《消费者权益保护法》", "三倍赔偿", "12315投诉"]
    },
    # ⑦ 刑法类（测试边界能力）
    {
        "id": "criminal_01",
        "category": "刑法",
        "type": "训练集外",
        "question": "被人打伤住院，对方除了民事赔偿，还有可能承担刑事责任吗？",
        "key_points": ["轻伤以上", "故意伤害罪", "刑事附带民事"]
    },
]

SYSTEM_PROMPT = "你是一个专业的中国法律顾问，请根据用户的法律问题，给出准确、专业、有法律依据的回答。"

MODEL_PATHS = {
    "base": "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct",
    "sft":  "/root/autodl-tmp/merged_models/law_qa_sft",
    "dpo":  "/root/autodl-tmp/merged_models/law_qa_dpo",
}

def generate_answer(model, tokenizer, question, system_prompt, max_new_tokens=512):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    output_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "sft", "dpo"], required=True)
    args = parser.parse_args()
    
    model_path = MODEL_PATHS[args.model]
    print(f"\n{'='*60}")
    print(f"加载模型: {args.model.upper()} → {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("模型加载完成 ✅")
    
    results = []
    for case in TEST_CASES:
        print(f"\n[{case['id']}] {case['category']}（{case['type']}）")
        print(f"问题: {case['question']}")
        answer = generate_answer(model, tokenizer, case["question"], SYSTEM_PROMPT)
        print(f"回答: {answer[:200]}{'...' if len(answer)>200 else ''}")
        
        # 检查关键知识点覆盖情况
        covered = [kp for kp in case["key_points"] if kp in answer]
        coverage = len(covered) / len(case["key_points"])
        print(f"关键点覆盖: {len(covered)}/{len(case['key_points'])} ({coverage*100:.0f}%) → {covered}")
        
        results.append({
            "id": case["id"],
            "category": case["category"],
            "type": case["type"],
            "question": case["question"],
            "answer": answer,
            "key_points": case["key_points"],
            "covered_points": covered,
            "coverage_rate": coverage
        })
    
    # 保存结果
    os.makedirs("eval_results", exist_ok=True)
    output_file = f"eval_results/domain_eval_{args.model}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印汇总
    avg_coverage = sum(r["coverage_rate"] for r in results) / len(results)
    print(f"\n{'='*60}")
    print(f"模型: {args.model.upper()}")
    print(f"平均关键点覆盖率: {avg_coverage*100:.1f}%")
    print(f"结果已保存: {output_file}")

if __name__ == "__main__":
    main()