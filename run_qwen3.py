from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 使用本地绝对路径，避免联网检查触发 ConnectError
model_name = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True
)

print(f"Model loaded with dtype: {model.dtype}")
print(f"Model device: {model.device}")
if hasattr(model, "hf_device_map"):
    print(f"Model device map: {model.hf_device_map}")

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    # enable_thinking=True # Qwen3-0.6B 可能不支持 thinking mode，先注释掉
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512  # 32768 太长了，演示用 512 即可
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# parsing thinking content (Qwen3-0.6B 可能不输出 thinking content，这里做简单处理)
content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

print("content:", content)
