import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from bpe_tokenizer import BPETokenizer

# 1. 定义 Dataset
# Dataset 负责存储数据，并提供“取出一个样本”的逻辑
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        # 告诉 DataLoader 数据集有多大
        return len(self.texts)

    def __getitem__(self, idx):
        # 告诉 DataLoader 第 idx 个样本长什么样
        text = self.texts[idx]
        # 使用我们的 BPE Tokenizer 编码
        # 注意：这里我们把它转成 Tensor
        encoded_ids = self.tokenizer.encode(text)
        return torch.tensor(encoded_ids, dtype=torch.long)

# 2. 准备数据和 Tokenizer
texts = [
    "Hello world",
    "This is a test",
    "BPE is cool",
    "PyTorch Dataset and DataLoader are powerful tools for deep learning"
]

# 初始化并训练一个简单的 Tokenizer (为了演示，我们就用这些文本训练)
tokenizer = BPETokenizer()
tokenizer.train(" ".join(texts), vocab_size=270) # 小词表

# 实例化 Dataset
dataset = TextDataset(texts, tokenizer)

# 3. 定义 Collate Function
# 因为每个句子的长度不一样，DataLoader 默认无法把它们堆叠成一个 Batch
# 我们需要一个函数来处理“如何把多个样本拼在一起”
def collate_fn(batch):
    # batch 是一个列表，包含了 batch_size 个 __getitem__ 的返回值
    # 即 [tensor([1, 2]), tensor([3, 4, 5]), ...]
    
    # 使用 pad_sequence 自动填充，使得长度一致
    # padding_value=0 (假设 0 是 pad token，虽然我们的 tokenizer 没专门定义，这里暂且用 0)
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
    return padded_batch

# 4. 定义 DataLoader
# DataLoader 负责：批量加载、打乱顺序、多进程读取等
dataloader = DataLoader(
    dataset,
    batch_size=2,      # 每次取 2 个样本
    shuffle=True,      # 打乱顺序
    collate_fn=collate_fn # 使用我们定义的填充函数
)

# 5. 迭代 DataLoader
print("Starting iteration...")
for batch_idx, batch in enumerate(dataloader):
    print(f"\n--- Batch {batch_idx} ---")
    print(f"Shape: {batch.shape}")
    print(f"Data:\n{batch}")
    
    # 简单的解码演示 (只解码第一个样本，忽略 padding 0)
    # 注意：我们的 tokenizer decode 需要 list[int]
    first_sample_ids = batch[0].tolist()
    # 过滤掉 padding (0)
    valid_ids = [i for i in first_sample_ids if i != 0]
    print(f"Decoded (1st sample): '{tokenizer.decode(valid_ids)}'")
