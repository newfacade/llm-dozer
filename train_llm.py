import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# Imports from local files
from torch_train.torch_tokenizer import BPETokenizer
from torch_train.torch_model import TransformerModel
from torch_train.torch_dataset import PretrainDataset
from torch.cuda.amp import autocast, GradScaler

# ==========================================
# Configuration
# ==========================================
CONFIG = {
    # Data
    "parquet_path": "data/wikitext-103-raw-v1-validation.parquet", 
    "tokenizer_path": "wiki-tokenizer-1.json",
    "seq_len": 128,          # 增加 Context Window 到 128 (8G 显存可以尝试 256 或 512，但先保守点)
    "batch_size": 16,        # 增加 Batch Size
    
    # Model (Small size ~ 10M parameters for demo)
    # GPT-2 Small is ~124M (12 layers, 768 d_model). 
    # 0.5B is too large for 8G VRAM training without optimizations.
    # Let's try a "Mini" model that fits comfortably.
    "d_model": 256,          
    "n_head": 8,
    "d_hidden": 1024,        
    "n_layer": 6,            
    "dropout": 0.1,
    
    # Training
    "lr": 5e-4,
    "epochs": 2,             
    "log_interval": 10,
    "save_path": "llm_checkpoint.pt",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp": True,         # Enable Automatic Mixed Precision
}

def train():
    print(f"Using device: {CONFIG['device']}")
    
    # ------------------------------------------------------------------
    # 1. Load Tokenizer
    # ------------------------------------------------------------------
    print("Loading Tokenizer...")
    tokenizer = BPETokenizer()
    if os.path.exists(CONFIG["tokenizer_path"]):
        tokenizer.load(CONFIG["tokenizer_path"])
        print(f"Tokenizer loaded. Vocab size: {len(tokenizer.vocab)}")
    else:
        raise FileNotFoundError(f"Tokenizer file not found at {CONFIG['tokenizer_path']}")

    vocab_size = len(tokenizer.vocab)
    pad_id = tokenizer.special_tokens.get("<PAD>", 0)

    # ------------------------------------------------------------------
    # 2. Load Data & Create Dataset
    # ------------------------------------------------------------------
    print(f"Loading data from {CONFIG['parquet_path']}...")
    # 这里我们直接传入文件路径列表，而不是读取后的文本列表
    # 如果有多个文件，可以扩展这个列表
    file_paths = [CONFIG["parquet_path"]] 
    
    print("Initializing Dataset...")
    # Request seq_len + 1 to handle input/target shift
    dataset = PretrainDataset(file_paths, tokenizer, seq_len=CONFIG["seq_len"] + 1)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=2
    )

    # ------------------------------------------------------------------
    # 3. Initialize Model
    # ------------------------------------------------------------------
    print("Initializing Model...")
    model = TransformerModel(
        d_model=CONFIG["d_model"],
        n_head=CONFIG["n_head"],
        d_hidden=CONFIG["d_hidden"],
        n_layer=CONFIG["n_layer"],
        vocab_size=vocab_size,
        max_seq_len=CONFIG["seq_len"] + 1, # Should handle at least this
        dropout=CONFIG["dropout"]
    ).to(CONFIG["device"])

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # ------------------------------------------------------------------
    # 4. Training Loop
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id) # Ignore padding in loss
    scaler = GradScaler(enabled=CONFIG["use_amp"])

    model.train()
    
    print("Starting training...")
    for epoch in range(CONFIG["epochs"]):
        total_loss = 0
        # IterableDataset doesn't implement __len__, so we can't use total=len(dataloader)
        progress_bar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in progress_bar:
            batch = batch.to(CONFIG["device"])
            
            # Split into Input and Target
            # Input:  [x0, x1, ..., xN-1]
            # Target: [x1, x2, ..., xN]
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]
            
            # Generate position_ids
            B, T = input_ids.shape
            position_ids = torch.arange(T, device=CONFIG["device"]).unsqueeze(0).expand(B, T)
            
            # Forward
            optimizer.zero_grad()
            
            with autocast(enabled=CONFIG["use_amp"]):
                logits = model(input_ids, position_ids) # [B, T, Vocab]
                
                # Reshape for Loss
                # logits: [B*T, Vocab], target: [B*T]
                loss = criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))
            
            # Backward
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            # Logging
            if batch_idx % CONFIG["log_interval"] == 0:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Since we can't know total length easily, avg_loss calculation might need adjustment
        # or we just count batches
        num_batches = batch_idx + 1
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), CONFIG["save_path"])
        print(f"Checkpoint saved to {CONFIG['save_path']}")

if __name__ == "__main__":
    train()
