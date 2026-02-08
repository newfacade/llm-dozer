import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, get_worker_info
import math
import random
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from torch_train.torch_tokenizer import BPETokenizer


class PretrainDataset(Dataset):
    def __init__(self, file_paths, tokenizer, seq_len, text_col='page'):
        """
        :param file_paths: Parquet 文件路径列表
        :param tokenizer: 分词器 (需要包含 <EOS> 和 <PAD>)
        :param seq_len: 训练序列长度 (Context Window)
        """
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
            
        print(f"Processing {len(file_paths)} files...")
        
        # 1. 读取并编码所有文本
        texts = []
        for file_path in tqdm(file_paths, desc="Processing files"):
            try:
                df = pd.read_parquet(file_path)
                if text_col in df.columns:
                    texts.extend(df[text_col].dropna().tolist())
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        # 为了增加随机性，可以在拼接前打乱文本顺序
        random.shuffle(texts)
        
        all_token_ids = []
        for text in tqdm(texts, desc="Tokenizing texts"):
            ids = tokenizer.encode(text)
            all_token_ids.extend(ids)
            # 每条数据后加 EOS
            all_token_ids.append(self.eos_id)
            
        # 转为 Tensor 存储
        self.data = torch.tensor(all_token_ids, dtype=torch.long)
        
        # 计算样本总数
        self.num_samples = math.ceil(len(self.data) / self.seq_len)
        print(f"Total tokens: {len(self.data)}. Total samples (seq_len={seq_len}): {self.num_samples}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 计算切片范围
        start = idx * self.seq_len
        end = min(start + self.seq_len, len(self.data))
        
        chunk = self.data[start:end]
        
        # 如果长度不足 seq_len，进行 Padding
        if len(chunk) < self.seq_len:
            pad_len = self.seq_len - len(chunk)
            padding = torch.full((pad_len,), self.pad_id, dtype=torch.long)
            chunk = torch.cat([chunk, padding])
            
        return chunk


class IterablePretrainDataset(IterableDataset):
    def __init__(self, file_paths, tokenizer, seq_len, text_col='page'):
        """
        :param file_paths: Parquet 文件路径列表
        :param tokenizer: 分词器 (需要包含 <EOS> 和 <PAD>)
        :param seq_len: 训练序列长度 (Context Window)
        :param text_col: 文本列名
        """
        self.file_paths = file_paths
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.text_col = text_col
        
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
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            my_files = [f for i, f in enumerate(self.file_paths) if i % num_workers == worker_id]
            
        buffer = []
        
        for file_path in my_files:
            try:
                # 使用 pyarrow 流式读取 Parquet
                parquet_file = pq.ParquetFile(file_path)
                
                # 每次读取一个行组 (Row Group) 或者指定 batch_size
                for batch in parquet_file.iter_batches(batch_size=1000):
                    df = batch.to_pandas()
                    
                    if self.text_col in df.columns:
                        texts = df[self.text_col].dropna().tolist()
                        for text in texts:
                            ids = self.tokenizer.encode(text)
                            ids.append(self.eos_id) # 添加 EOS
                            buffer.extend(ids)
                            
                            # 当 buffer 足够切分时，yield 出来
                            while len(buffer) >= self.seq_len:
                                yield torch.tensor(buffer[:self.seq_len], dtype=torch.long)
                                buffer = buffer[self.seq_len:]
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue
                
        # 处理剩余的 buffer (如果不为空)，进行 Padding
        if len(buffer) > 0:
            pad_len = self.seq_len - len(buffer)
            padding = [self.pad_id] * pad_len
            buffer.extend(padding)
            yield torch.tensor(buffer, dtype=torch.long)


if __name__ == "__main__":
    tokenizer = BPETokenizer()
    tokenizer.load('wiki-tokenizer-1.json')

    file_paths = ['data/wikitext-103-raw-v1-validation.parquet']
    dataset = PretrainDataset(file_paths, tokenizer, seq_len=256, text_col='page')
    dataloader = DataLoader(
        dataset,
        batch_size=2,      # 每次取 2 个句子
        shuffle=True,      # 打乱顺序
    )

    print("Starting iteration...")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n--- Batch {batch_idx} ---")
        print(f"Shape: {batch.shape}")
        # print(f"Data:\n{batch}")
        
        # 简单的解码演示 (只解码第一个样本，忽略 padding 0)
        # 注意：我们的 tokenizer decode 需要 list[int]
        first_sample_ids = batch[0].tolist()
        # 过滤掉 padding (0)
        valid_ids = [i for i in first_sample_ids if i != 0]
        print(f"Decoded (1st sample): '{tokenizer.decode(valid_ids)}'")
        break

