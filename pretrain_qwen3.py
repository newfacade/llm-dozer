import torch
import os
from transformers import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# ==========================================
# 0. 环境准备 (Offline Mode)
# ==========================================
# 强制使用本地缓存，避免联网报错
os.environ["HF_HUB_OFFLINE"] = "1"

def train_qwen3_from_scratch():
    print("=== 1. 定义模型配置 (Configuration) ===")
    local_model_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
    )
    if not os.path.exists(local_model_path):
        raise FileNotFoundError(
            f"找不到本地 Qwen3 模型目录：{local_model_path}\n"
            "请先确保你已经把 Qwen3-0.6B 缓存到本地（或把 local_model_path 改成你自己的 snapshot 绝对路径）。"
        )

    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Qwen3Config 继承自 PretrainedConfig
    # 这里定义一个“Qwen3-Nano”用于跑通从 0 开始的预训练流程（随机初始化 + Causal LM）
    # 注意：这不是复现 Qwen3-0.6B 的规模，只是同架构缩小版。
    hidden_size = 128
    num_attention_heads = 2
    head_dim = hidden_size // num_attention_heads
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=2,
        head_dim=head_dim,
        hidden_act="silu",
        max_position_embeddings=256,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(f"Model Config: {config}")

    print("\n=== 2. 初始化模型 (Initialization) ===")
    # 使用随机初始化（因为是从头预训练，不是微调）
    model = Qwen3ForCausalLM(config)
    
    # 打印参数量
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Qwen3-Nano Parameters: {model_size/1000**2:.2f}M")

    print("\n=== 3. 准备数据 (Data Preparation) ===")
    # 这是一个演示用的 Dummy Dataset
    # 实际预训练时，你应该加载大规模文本 (e.g., wikitext, c4)
    # 这里的文本尽量覆盖词表里的一些 token
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        "I love machine learning and transformers.",
        "Qwen is a powerful language model developed by Alibaba Cloud.",
        "Hugging Face provides great tools for NLP."
    ] * 100 # 重复多次以构成一个 epoch

    # 数据处理函数
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)

    raw_dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
    
    # Data Collator: 负责把数据拼成 batch，并处理 labels
    # 对于 Causal LM，labels = input_ids (shifted inside the model)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("\n=== 4. 训练设置 (Training Arguments) ===")
    args = TrainingArguments(
        output_dir="./qwen3_nano_pretrain",
        # overwrite_output_dir=True,
        max_steps=20,
        per_device_train_batch_size=2,
        save_steps=50,
        logging_steps=10,
        learning_rate=1e-4,
        weight_decay=0.01,
        # bf16=True, # 如果你的 GPU 支持 BF16 (Ampere+)，建议开启
        # fp16=False,
        use_cpu=not torch.cuda.is_available(),
        report_to="none"
    )

    print("\n=== 5. 开始训练 (Start Training) ===")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    
    print("\n=== 6. 保存模型 (Save Model) ===")
    trainer.save_model("./qwen3_nano_final")
    tokenizer.save_pretrained("./qwen3_nano_final")
    print("Done! Model saved to ./qwen3_nano_final")

if __name__ == "__main__":
    train_qwen3_from_scratch()
