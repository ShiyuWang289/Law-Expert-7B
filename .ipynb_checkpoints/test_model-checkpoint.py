import json

model_info_path = "data/model_info.json"

# 读取现有配置
with open(model_info_path, 'r', encoding='utf-8') as f:
    model_info = json.load(f)

# 添加 Qwen2.5-7B-Instruct 配置
if "qwen2.5-7b-instruct" not in model_info:
    model_info["qwen2.5-7b-instruct"] = {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "local_path": "models/Qwen/Qwen2___5-7B-Instruct",
        "template": "qwen"
    }
    
    # 保存
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print("✅ 模型配置已添加")
else:
    print("ℹ️ 模型配置已存在")