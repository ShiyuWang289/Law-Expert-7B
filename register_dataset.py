# register_dataset.py
import json

config_path = "data/dataset_info.json"

# 读取现有配置
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# 注册 Alpaca 格式数据集
config["law_qa_alpaca"] = {
    "file_name": "law_qa_alpaca.json",
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "system": "system"
    }
}

# 注册 ShareGPT 格式数据集
config["law_qa_sharegpt"] = {
    "file_name": "law_qa_sharegpt.json",
    "formatting": "sharegpt",
    "columns": {
        "messages": "conversations",
        "system": "system"
    }
}

# 写回配置文件
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print("✅ 数据集注册成功！")
print(f"  - law_qa_alpaca （Alpaca 格式）")
print(f"  - law_qa_sharegpt （ShareGPT 格式）")

# 验证
with open(config_path, 'r', encoding='utf-8') as f:
    verify = json.load(f)
    assert "law_qa_alpaca" in verify, "注册失败！"
    assert "law_qa_sharegpt" in verify, "注册失败！"
    print(f"\n📋 当前 dataset_info.json 共注册 {len(verify)} 个数据集")