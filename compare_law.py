import json

models = {
    "Base": "/root/autodl-tmp/eval_results/base/results.json",
    "SFT": "/root/autodl-tmp/eval_results/sft/results.json",
    "DPO": "/root/autodl-tmp/eval_results/dpo/results.json"
}

# 法律相关科目
law_subjects = ["law", "legal_professional"]

# 加载正确答案（从 ceval 数据集）
from datasets import load_dataset

print("加载 ceval 正确答案...")
correct_answers = {}

for subject in law_subjects:
    try:
        ds = load_dataset(
            path="ceval/ceval-exam",
            name=subject,
            trust_remote_code=True,
            split="validation"
        )
        correct_answers[subject] = {str(i): item['answer'] for i, item in enumerate(ds)}
    except Exception as e:
        print(f"  {subject}: 加载失败 {e}")

print("\n" + "=" * 60)
print(f"{'科目':<25} {'Base':>8} {'SFT':>8} {'DPO':>8}")
print("=" * 60)

model_law_acc = {}

for model_name, result_file in models.items():
    with open(result_file, 'r') as f:
        predictions = json.load(f)
    
    total_correct = 0
    total_count = 0
    
    for subject in law_subjects:
        if subject not in predictions or subject not in correct_answers:
            continue
        
        pred = predictions[subject]
        gold = correct_answers[subject]
        
        correct = sum(1 for k in pred if k in gold and pred[k] == gold[k])
        total = len(gold)
        acc = correct / total * 100
        
        total_correct += correct
        total_count += total
        
        if model_name == "Base":
            print(f"{subject:<25} ", end="")
        
    model_law_acc[model_name] = total_correct / total_count * 100 if total_count > 0 else 0

# 打印各科目准确率
for subject in law_subjects:
    if subject not in correct_answers:
        continue
    
    scores = []
    for model_name in ["Base", "SFT", "DPO"]:
        with open(models[model_name], 'r') as f:
            pred = json.load(f)
        
        if subject in pred and subject in correct_answers:
            gold = correct_answers[subject]
            correct = sum(1 for k in pred[subject] if k in gold and pred[subject][k] == gold[k])
            acc = correct / len(gold) * 100
            scores.append(f"{acc:.1f}%")
        else:
            scores.append("N/A")
    
    print(f"{subject:<25} {scores[0]:>8} {scores[1]:>8} {scores[2]:>8}")

print("-" * 60)
print(f"{'法律平均':<25} ", end="")
for model_name in ["Base", "SFT", "DPO"]:
    print(f"{model_law_acc.get(model_name, 0):>7.1f}%", end=" ")
print()

print("\n" + "=" * 60)
print("\n通用能力对比：")
print(f"  Base: 78.83%")
print(f"  SFT:  79.42% (+0.59)")
print(f"  DPO:  79.05% (+0.22)")

