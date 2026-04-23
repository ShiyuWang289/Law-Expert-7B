# register_dpo_dataset.py
import json

config_path = "data/dataset_info.json"

with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# 注册 DPO 偏好数据集
config["law_qa_dpo"] = {
    "file_name": "law_qa_dpo.json",
    "ranking": True,                    # ★ 关键！标记这是偏好数据集
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "chosen": "chosen",
        "rejected": "rejected"
    }
}

with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print("✅ DPO 数据集注册成功！")

# 验证
with open(config_path, 'r', encoding='utf-8') as f:
    verify = json.load(f)
    entry = verify["law_qa_dpo"]
    assert entry["ranking"] == True, "ranking 必须为 True！"
    assert "chosen" in entry["columns"], "缺少 chosen 映射！"
    assert "rejected" in entry["columns"], "缺少 rejected 映射！"
    print(f"   配置: {json.dumps(entry, ensure_ascii=False)}")