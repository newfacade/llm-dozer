import torch
from torch.utils.data import IterableDataset, DataLoader
import os
import math

# ==========================================
# 1. 模拟生成几个“大”文件
# ==========================================
def create_dummy_large_files():
    # 模拟 4 个大文件
    filenames = []
    for i in range(4):
        fname = f"huge_corpus_part_{i}.txt"
        with open(fname, "w", encoding="utf-8") as f:
            for j in range(100): # 每个文件 100 行
                f.write(f"File_{i} Row_{j}: This is a sample sentence from a large dataset.\n")
        filenames.append(fname)
    return filenames

# ==========================================
# 2. 定义 IterableDataset
# ==========================================
class LargeTextIterableDataset(IterableDataset):
    def __init__(self, file_paths, tokenizer=None):
        self.file_paths = file_paths
        self.tokenizer = tokenizer

    def parse_file(self, file_path):
        """生成器：流式读取单个文件，不一次性加载到内存"""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 如果有 tokenizer 就做处理，否则直接返回文本
                if self.tokenizer:
                    # 简单演示：只返回长度作为 Tensor
                    yield torch.tensor(len(line), dtype=torch.long)
                else:
                    yield line

    def __iter__(self):
        # 关键点：处理多进程 DataLoader (num_workers > 0)
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # 单进程模式：直接读取所有文件
            my_files = self.file_paths
            print(f"[Main Process] Reading all {len(my_files)} files")
        else:
            # 多进程模式：需要切分任务
            # 策略：按文件分配。Worker 0 读文件 [0, 2, ...], Worker 1 读文件 [1, 3, ...]
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
            my_files = [
                f for i, f in enumerate(self.file_paths) 
                if i % num_workers == worker_id
            ]
            print(f"[Worker {worker_id}] Assigned {len(my_files)} files: {my_files}")

        # 开始流式迭代分配给我的文件
        for file_path in my_files:
            yield from self.parse_file(file_path)

# ==========================================
# 3. 演示
# ==========================================
if __name__ == "__main__":
    # 1. 准备数据
    files = create_dummy_large_files()
    print(f"Created files: {files}")

    # 2. 实例化 Dataset
    dataset = LargeTextIterableDataset(files)

    # 3. 实例化 DataLoader
    # 注意：对于 IterableDataset，shuffle=True 是不支持的（会报错）
    # 如果需要混洗，必须在 Dataset 内部维护一个 buffer 进行局部混洗
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        num_workers=2 # 开启 2 个 worker 并行读取
    )

    print("\nStarting iteration with DataLoader (num_workers=2)...")
    for batch_idx, batch in enumerate(dataloader):
        # 简单打印前几个 batch
        if batch_idx < 5:
            print(f"Batch {batch_idx}: {batch}")
        else:
            break
            
    print("\nDone. Note how data from different files is interleaved because of multiple workers.")
    
    # 清理文件
    for f in files:
        if os.path.exists(f):
            os.remove(f)
