import torch
import torch.nn as nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    LlamaTokenizer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import os

# ==========================================
# 1. 定义自定义模型配置 (Configuration)
# ==========================================
class MyCustomLLMConfig(PretrainedConfig):
    model_type = "my_custom_llm"
    
    def __init__(
        self,
        vocab_size=32000,
        n_embd=768,      # Embedding Dimension
        n_layer=12,      # Number of Layers
        n_head=12,       # Number of Attention Heads
        n_positions=1024,# Max Sequence Length
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_positions = n_positions
        self.dropout = dropout

# ==========================================
# 2. 定义自定义模型结构 (Model Structure)
# ==========================================
# 继承 PreTrainedModel 以兼容 Hugging Face Trainer
class MyCustomLLM(PreTrainedModel):
    config_class = MyCustomLLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # --- 模型核心组件 ---
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.n_positions, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # 使用 PyTorch 标准 TransformerEncoder 作为示例
        # 实际生产中你可能会手写 DecoderBlock (如 Llama 的 RMSNorm + SwiGLU + RoPE)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=4 * config.n_embd,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True # Pre-Norm 结构更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layer)
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 权重初始化
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        b, t = input_ids.size()
        
        # 生成位置编码
        pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device)
        
        # Embedding
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        x = self.dropout(x)
        
        # Causal Mask (防止看到未来)
        # PyTorch Transformer 需要 mask 形状为 (Seq_Len, Seq_Len)
        # float mask: 0.0 for visible, -inf for invisible
        mask = nn.Transformer.generate_square_subsequent_mask(t).to(input_ids.device)
        
        # Transformer Forward
        # 注意：这里如果传 attention_mask (padding mask)，需要处理形状
        # 简单起见，这里只演示 Causal Mask
        x = self.transformer(x, mask=mask, is_causal=True)
        
        # Head
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        # Trainer 需要返回 (loss, logits) 或者 ModelOutput
        # 只要包含 loss 即可自动进行反向传播
        return {"loss": loss, "logits": logits}

# ==========================================
# 3. 准备数据和 Tokenizer
# ==========================================
def prepare_data_and_tokenizer():
    # 尝试加载上一轮生成的 Llama Tokenizer
    tokenizer_path = "llama_tokenizer_demo.model"
    if os.path.exists(tokenizer_path):
        print(f"Loading local tokenizer from {tokenizer_path}")
        tokenizer = LlamaTokenizer(vocab_file=tokenizer_path)
        # LlamaTokenizer 默认没有 pad_token，训练时需要指定
        tokenizer.pad_token = tokenizer.eos_token 
    else:
        print("Local tokenizer not found, using GPT-2 tokenizer as fallback")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

    # 构造一些假数据
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        "Artificial Intelligence is the future of technology.",
        "Python is a great programming language for AI.",
        "Hugging Face makes NLP easy and accessible."
    ] * 20 # 复制多次凑够 batch
    
    dataset = Dataset.from_dict({"text": texts})
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=32
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # 移除原始文本列，只保留 input_ids 和 attention_mask
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    
    return tokenizer, tokenized_datasets

# ==========================================
# 4. 主训练流程
# ==========================================
def main():
    # 1. 准备组件
    tokenizer, dataset = prepare_data_and_tokenizer()
    
    # 2. 初始化配置和模型
    # 这里的参数应该根据你的显存大小调整
    config = MyCustomLLMConfig(
        vocab_size=len(tokenizer),
        n_embd=256,    # 小一点方便演示 (Demo size)
        n_layer=4,     # 4层
        n_head=4,      # 4头
        n_positions=32 # 序列长度
    )
    
    model = MyCustomLLM(config)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 3. 设置训练参数
    training_args = TrainingArguments(
        output_dir="./custom_llm_checkpoints",
        # overwrite_output_dir=True, # 暂时注释掉，避免 transformers 5.x 报错
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=50,
        logging_steps=10,
        learning_rate=1e-3,
        weight_decay=0.01,
        # fp16=torch.cuda.is_available(), # 如果有 GPU 推荐开启
        report_to="none", # 不上传 wandb
        use_cpu=not torch.cuda.is_available()
    )
    
    # 4. 初始化 Trainer
    # DataCollatorForLanguageModeling 会自动处理 labels (Shift)
    # mlm=False 表示是 Causal LM (Next Token Prediction)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # 5. 开始训练
    print("Starting training...")
    trainer.train()
    
    # 6. 保存模型
    trainer.save_model("./custom_llm_final")
    print("Training finished and model saved to ./custom_llm_final")

if __name__ == "__main__":
    main()
