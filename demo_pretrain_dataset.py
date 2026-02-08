import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import math
import random
from functools import partial
import pyarrow.parquet as pq
import pandas as pd
from bpe_tokenizer import BPETokenizer

class PretrainDataset(IterableDataset):
    def __init__(self, file_paths, tokenizer, seq_len):
        """
        :param file_paths: Parquet 文件路径列表
        :param tokenizer: 分词器 (需要包含 <EOS> 和 <PAD>)
        :param seq_len: 训练序列长度 (Context Window)
        """
        self.file_paths = file_paths
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        
        # 获取特殊 token ID
        self.eos_id = tokenizer.special_tokens.get("<EOS>")
        self.pad_id = tokenizer.special_tokens.get("<PAD>")
        
        if self.eos_id is None:
            raise ValueError("Tokenizer must have <EOS> defined for pretraining.")
        if self.pad_id is None:
            print("Warning: <PAD> not found in tokenizer. Using 0 as pad_id.")
            self.pad_id = 0
            
    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # 单进程模式，处理所有文件
            my_files = self.file_paths
        else:
            # 多进程模式，按 worker_id 分配文件
            # 简单的 stride 分配: file[i] 分给 worker[i % num_workers]
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            my_files = [f for i, f in enumerate(self.file_paths) if i % num_workers == worker_id]
            
        # 可以在文件级别 shuffle (如果需要的话)
        # random.shuffle(my_files)
        
        buffer = []
        
        for file_path in my_files:
            try:
                # 使用 pyarrow 流式读取 Parquet
                parquet_file = pq.ParquetFile(file_path)
                
                # 每次读取一个行组 (Row Group) 或者指定 batch_size
                # batch_size=1000 行
                for batch in parquet_file.iter_batches(batch_size=1000):
                    df = batch.to_pandas()
                    
                    # 查找文本列
                    if "page" in df.columns:
                        texts = df["page"].dropna().tolist()
                    elif "text" in df.columns:
                        texts = df["text"].dropna().tolist()
                    else:
                        continue # 跳过没有文本列的 batch
                        
                    for text in texts:
                        # Tokenize
                        ids = self.tokenizer.encode(text)
                        ids.append(self.eos_id) # 添加 EOS
                        buffer.extend(ids)
                        
                        # 当 buffer 足够切分时，yield 出来
                        while len(buffer) >= self.seq_len:
                            chunk = buffer[:self.seq_len]
                            buffer = buffer[self.seq_len:]
                            yield torch.tensor(chunk, dtype=torch.long)
                            
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue
                
        # 处理剩余的 buffer (如果不为空)
        # 这里选择 yield 出来，交给 collate_fn 去 padding
        if len(buffer) > 0:
            yield torch.tensor(buffer, dtype=torch.long)

def collate_fn(batch, pad_id=0, min_len=None):
    """
    自定义的 collate_fn
    :param batch: list of tensors
    :param pad_id: padding 的 token id
    :param min_len: 强制最小长度 (通常是 seq_len)，如果不指定则按 batch 中最长的 pad
    """
    if min_len is not None:
        padded_batch = []
        for item in batch:
            if len(item) < min_len:
                # 计算需要补多少个 PAD
                pad_len = min_len - len(item)
                padding = torch.full((pad_len,), pad_id, dtype=torch.long)
                item = torch.cat([item, padding])
            padded_batch.append(item)
        return torch.stack(padded_batch)
    
    from torch.nn.utils.rnn import pad_sequence
    return pad_sequence(batch, batch_first=True, padding_value=pad_id)

# ==========================================
# 演示代码
# ==========================================
if __name__ == "__main__":
    import os
    
    # 1. 创建一个临时的 parquet 文件用于测试
    dummy_data = {
        "page": [
            "First long document about AI.",
            "Second document is about Python.",
            "Third one is very short.",
            "Fourth document is extremely long and contains many many words to test the splitting functionality of our dataset."
        ]
    }
    df = pd.DataFrame(dummy_data)
    test_parquet = "test_data.parquet"
    df.to_parquet(test_parquet)
    
    # 2. 训练 Tokenizer (确保包含 EOS 和 PAD)
    special_tokens = ["<EOS>", "<PAD>"]
    tokenizer = BPETokenizer()
    tokenizer.train(" ".join(dummy_data["page"]), vocab_size=300, special_tokens=special_tokens)
    
    print(f"Special Tokens: {tokenizer.special_tokens}")
    pad_id = tokenizer.special_tokens["<PAD>"]
    
    # 3. 实例化 IterableDataset
    # 传入文件列表
    seq_len = 10
    dataset = PretrainDataset([test_parquet], tokenizer, seq_len=seq_len)
    
    # 4. DataLoader
    # 注意: IterableDataset 不能设置 shuffle=True (shuffle 应该在 dataset 内部做)，
    # 并且 num_workers > 0 时需要测试多进程切分
    my_collate = partial(collate_fn, pad_id=pad_id, min_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=my_collate, num_workers=2)
    
    print("\n--- Iterating DataLoader ---")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"Shape: {batch.shape}")
        
        # 解码查看内容
        for i, sample in enumerate(batch):
            ids = sample.tolist()
            valid_ids = [idx for idx in ids if idx != pad_id]
            decoded = tokenizer.decode(valid_ids)
            print(f"  Sample {i} Decoded: '{decoded}'")
            
    # 清理临时文件
    if os.path.exists(test_parquet):
        os.remove(test_parquet)
