import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SimpleNextTokenModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # 1. Embedding 层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. MLP 层 (Multi-Layer Perceptron)
        # 简单结构: Linear -> Activation -> Linear
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),  # 也可以使用 ReLU
            nn.Linear(hidden_dim, embed_dim) # 投影回 embed_dim
        )
        
        # 3. 输出层 (Linear)
        # 将维度从 embed_dim 映射回 vocab_size
        self.output_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        """
        参数:
            input_ids: (batch_size, sequence_length)
        返回:
            logits: (batch_size, sequence_length, vocab_size)
        """
        # 1. Embedding
        # x shape: (batch_size, seq_len, embed_dim)
        x = self.embedding(input_ids)
        
        # 2. MLP
        # x shape: (batch_size, seq_len, embed_dim)
        x = self.mlp(x)
        
        # 3. 输出层
        # logits shape: (batch_size, seq_len, vocab_size)
        logits = self.output_head(x)
        
        return logits

def generate_dummy_data(num_batches, batch_size, seq_len, vocab_size):
    """生成简单的重复模式数据，方便观察 loss 下降"""
    data = []
    for _ in range(num_batches):
        # 随机生成数据
        batch = torch.randint(0, vocab_size, (batch_size, seq_len))
        data.append(batch)
    return data

if __name__ == "__main__":
    # 1. 设置随机种子，保证可复现性
    torch.manual_seed(42)

    # 自动检测设备
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"使用设备: {device}")

    # 2. 超参数
    vocab_size = 1000
    embed_dim = 64
    hidden_dim = 256
    seq_len = 16   # 序列长度
    batch_size = 32
    learning_rate = 1e-3
    num_batches = 100 # 训练步数
    
    # 3. 初始化模型并移动到设备
    model = SimpleNextTokenModel(vocab_size, embed_dim, hidden_dim).to(device)
    print("模型结构初始化完成。")
    
    # 4. 定义 Loss 和 Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 5. 准备虚拟数据
    # 注意：这里的 data 包含输入和目标
    # 假设我们有一个长序列，我们将其切分为 batch
    # 这里简化为直接生成 batch
    train_data = generate_dummy_data(num_batches, batch_size, seq_len + 1, vocab_size)
    
    print(f"开始训练，共 {num_batches} 个 batch...")
    model.train()
    
    for step, batch in enumerate(train_data):
        # 将数据移动到设备
        batch = batch.to(device)

        # 构造输入和目标 (Next Token Prediction)
        # 输入: 序列的前 N-1 个 token
        # 目标: 序列的后 N-1 个 token (即每个位置的下一个 token)
        input_ids = batch[:, :-1]  # (B, T)
        targets = batch[:, 1:]     # (B, T)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        logits = model(input_ids) # (B, T, V)
        
        # 计算 Loss
        # CrossEntropyLoss 需要 (N, C) 和 (N) 的输入，或者 (N, C, d1...) 和 (N, d1...)
        # 这里我们把 batch 和 time 维度展平
        # logits.view(-1, vocab_size) -> (B*T, V)
        # targets.view(-1) -> (B*T)
        loss = criterion(logits.view(-1, vocab_size), targets.reshape(-1))
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        if (step + 1) % 10 == 0:
            print(f"Step [{step+1}/{num_batches}], Loss: {loss.item():.4f}")

    print("训练结束！")
    
    # 简单验证一下
    test_input = train_data[0][:, :-1].to(device)
    with torch.no_grad():
        logits = model(test_input)
        preds = torch.argmax(logits, dim=-1)
        print("\n验证第一个 batch 的预测:")
        print(f"输入 shape: {test_input.shape}")
        print(f"预测 shape: {preds.shape}")
        # 这里只是随机数据，预测准确率不会高，主要是验证流程跑通
