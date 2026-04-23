# verify_merged.py
from transformers import AutoTokenizer
import os

for model_name, model_path in [
    ("SFT", "/root/autodl-tmp/merged_models/law_qa_sft"),
    ("DPO", "/root/autodl-tmp/merged_models/law_qa_dpo"),
]:
    # 检查关键文件
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    all_exist = all(os.path.exists(f"{model_path}/{f}") for f in required_files)
    
    # 检查有 safetensors 文件
    shard_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    
    # 加载 tokenizer 验证
    try:
        tok = AutoTokenizer.from_pretrained(model_path)
        tok_ok = True
    except Exception as e:
        tok_ok = False
    
    print(f"\n{'='*40}")
    print(f"模型: {model_name}")
    print(f"路径: {model_path}")
    print(f"关键文件: {'✅' if all_exist else '❌'}")
    print(f"权重分片: {len(shard_files)} 个 ({'✅' if shard_files else '❌'})")
    print(f"Tokenizer: {'✅' if tok_ok else '❌'}")