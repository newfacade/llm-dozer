import os
from transformers import AutoTokenizer

# Use the local path as discovered previously to avoid network issues
model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

messages = [
    {"role": "user", "content": "Hello, how are you?"}
]

print("--- 1. tokenize 参数对比 ---")
# tokenize=False: 返回拼接好的 prompt 字符串
text_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print(f"tokenize=False (type: {type(text_str)}):\n{text_str!r}")

# tokenize=True: 直接返回 token IDs 列表
text_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
print(f"tokenize=True (type: {type(text_ids)}):\n{text_ids}")


print("\n--- 2. add_generation_prompt 参数对比 (当 tokenize=False 时观察最明显) ---")
# add_generation_prompt=False: 只包含对话历史，不包含 AI 开始回复的引导词
prompt_no_gen = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print(f"add_generation_prompt=False:\n{prompt_no_gen!r}")

# add_generation_prompt=True: 自动追加 AI 回复的引导词（如 <|im_start|>assistant\n）
prompt_with_gen = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"add_generation_prompt=True:\n{prompt_with_gen!r}")


print("\n--- 3. return_tensors 参数对比 ---")
# 模拟一段输入文本
input_text = "Hello world"

# 默认 (return_tensors=None): 返回普通的 Python List
output_default = tokenizer(input_text)
print(f"默认 (return_tensors=None):\n类型: {type(output_default['input_ids'])}\n内容: {output_default['input_ids']}")

# return_tensors="pt": 返回 PyTorch Tensor
output_pt = tokenizer(input_text, return_tensors="pt")
print(f"return_tensors='pt':\n类型: {type(output_pt['input_ids'])}\n内容: {output_pt['input_ids']}")
