from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./models/Qwen/Qwen2___5-7B-Instruct"

print("加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("✅ Tokenizer 加载成功")

print("\n加载模型（这可能需要几分钟）...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("✅ 模型加载成功！")

print(f"\n模型信息:")
print(f"  - 参数量: {model.num_parameters() / 1e9:.2f}B")
print(f"  - 设备: {model.device}")

# 简单推理测试
print("\n测试推理...")
messages = [{"role": "user", "content": "你好"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"输入: 你好")
print(f"输出: {response}")
print("\n✅ 模型测试完成！")
