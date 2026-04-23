# inspect_data.py
import json

# 根据你的实际下载路径调整
file_path = "./raw_data/DISC-Law-SFT/DISC-Law-SFT-Pair-QA-released.jsonl"

# 查看前 3 条
with open(file_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        data = json.loads(line)
        print(f"=== 第 {i+1} 条 ===")
        print(f"字段: {list(data.keys())}")
        print(f"input: {data.get('input', '')[:80]}...")
        print(f"output: {data.get('output', '')[:80]}...")
        print()