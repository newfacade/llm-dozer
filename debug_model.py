
import math
import torch
import torch.nn as nn
from typing import Optional

# --- Copying code from notebook ---

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        ms = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def apply_rope_emb(x, cos, sin):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_size, max_position_embeddings, base):
        super().__init__()
        inv_freq = 1.0 / (base**(torch.arange(0, head_size, 2, dtype=torch.float) / head_size))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions):
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.transpose(1, 2)
        sin = sin.transpose(1, 2)
        return cos, sin

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        assert self.head_dim % 2 == 0
        self.wq = nn.Linear(d_model, n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_head * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_head * self.head_dim, bias=False)
        self.wo = nn.Linear(n_head * self.head_dim, d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.attn_dropout = dropout

    def forward(self, x, position_embeddings, attention_mask):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = self.q_norm(q.reshape(q.shape[0], q.shape[1], self.n_head, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.reshape(k.shape[0], k.shape[1], self.n_head, self.head_dim)).transpose(1, 2)
        v = v.reshape(v.shape[0], v.shape[1], self.n_head, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q = apply_rope_emb(q, cos, sin)
        k = apply_rope_emb(k, cos, sin)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn_weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attn_dropout, training=self.training)
        o = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        o = o.reshape(o.shape[0], o.shape[1], -1)
        return self.wo(o)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_hidden, bias=False)
        self.up_proj = nn.Linear(d_model, d_hidden, bias=False)
        self.down_proj = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_hidden, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = FeedForward(d_model, d_hidden)
        self.input_norm = RMSNorm(d_model)
        self.post_attention_norm = RMSNorm(d_model)

    def forward(self, x, position_embeddings, attention_mask):
        residual = x
        x = self.input_norm(x)
        x = self.attn(x, position_embeddings, attention_mask)
        x = x + residual
        residual = x
        x = self.post_attention_norm(x)
        x = self.ffn(x)
        x = x + residual
        return x

class TransformerModel(nn.Module):
    def __init__(self, d_model, n_head, d_hidden, n_layer, vocab_size, max_seq_len=8192, rope_theta=10000.0, dropout=0.0):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_head, d_hidden, dropout) for _ in range(n_layer)])
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.rotary_emb = RotaryEmbedding(d_model // n_head, max_seq_len, rope_theta)

    def forward(self, input_ids, position_ids):
        x = self.embeddings(input_ids)
        t = input_ids.shape[1]
        attention_mask = torch.triu(torch.full((t, t), float('-inf'), device=input_ids.device), diagonal=1).view(1, 1, t, t)
        position_embeddings = self.rotary_emb(position_ids)
        for layer in self.layers:
            x = layer(x=x, position_embeddings=position_embeddings, attention_mask=attention_mask)
        return self.output(self.norm(x))

# --- Testing ---
def test():
    d_model = 64
    n_head = 4
    d_hidden = 128
    n_layer = 2
    vocab_size = 100
    seq_len = 10
    batch_size = 2
    
    model = TransformerModel(d_model, n_head, d_hidden, n_layer, vocab_size)
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
    
    print("Running forward pass...")
    try:
        logits = model(input_ids, position_ids)
        print(f"Output shape: {logits.shape}")
        assert logits.shape == (batch_size, seq_len, vocab_size)
        print("Forward pass successful!")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
