from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset            # 新增导入
import torch
import os

model_path = "/root/autodl-tmp/merged_models/law_qa_dpo"
output_path = "/root/autodl-tmp/merged_models/law_qa_dpo_awq"

# 法律领域校准数据
CALIBRATION_DATA = [
    "根据《劳动合同法》第四十六条，劳动合同期满用人单位不续签应当支付经济补偿。",
    "根据《民法典》第七百二十二条，租赁期限内出租人不得解除合同。",
    "根据《消费者权益保护法》第二十五条，网购商品享有七日无理由退货权。",
    "用人单位违法解除劳动合同的，应当按经济补偿标准的二倍支付赔偿金。",
    "劳动者在工作时间和工作场所内因工作原因受到事故伤害的，应认定为工伤。",
    "离婚时夫妻共同财产由双方协议处理，协议不成的由法院判决。",
    "朋友借钱不还，可向对方住所地法院提起民事诉讼，注意诉讼时效为三年。",
    "交通事故责任认定不服，可在收到认定书三日内向上级交管部门申请复核。",
]

print("[1/3] 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("[2/3] 准备校准数据...")
# 将 list 转换为 Dataset 对象，列名必须为 "text"
dataset = Dataset.from_dict({"text": CALIBRATION_DATA})

# 可选：打印列名以确认格式
print("数据集列名:", dataset.column_names)

# AWQ INT4 量化配置
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W4A16",         # W4：权重 INT4，A16：激活 bf16
    ignore=["lm_head"],     # 输出层不量化，保证生成质量
)

print("[3/3] 执行量化（约 10~20 分钟）...")
oneshot(
    model=model,
    dataset=dataset,                       # 传入 Dataset 对象
    tokenizer=tokenizer,                   # 需要 tokenizer 以便对文本编码
    recipe=recipe,
    max_seq_length=512,
    num_calibration_samples=len(CALIBRATION_DATA),
    output_dir=output_path,
)

tokenizer.save_pretrained(output_path)

print(f"✅ 量化完成，保存至: {output_path}")

# 查看文件大小
total = sum(
    os.path.getsize(os.path.join(output_path, f))
    for f in os.listdir(output_path)
    if os.path.isfile(os.path.join(output_path, f))
)
print(f"量化后模型大小: {total/1024**3:.1f} GB")