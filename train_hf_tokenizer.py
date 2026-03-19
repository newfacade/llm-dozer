from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece

# 准备一点简单的训练数据
data = [
    "Hello, how are you?",
    "I am fine, thank you.",
    "This is a test sentence for training tokenizer.",
    "Hugging Face tokenizers library is very fast and powerful."
]

# 为了演示，我们需要把这些文本写入文件，因为 trainers 通常接受文件路径或迭代器
with open("train_data.txt", "w", encoding="utf-8") as f:
    for line in data:
        f.write(line + "\n")

print("=== 1. 训练 BPE Tokenizer (最常用，GPT-2/RoBERTa/Llama) ===")
# 初始化一个空的 BPE 模型
tokenizer_bpe = Tokenizer(BPE(unk_token="[UNK]"))

# 设置 Pre-tokenizer (预分词器)：先把句子切成词，通常用空格切分
# ByteLevel 是 GPT-2/3 用的，这里为了简单演示用 Whitespace
tokenizer_bpe.pre_tokenizer = pre_tokenizers.Whitespace()

# 设置 Trainer (训练器)
trainer_bpe = trainers.BpeTrainer(
    vocab_size=100,  # 为了演示设很小
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

# 训练
tokenizer_bpe.train(["train_data.txt"], trainer_bpe)

# 测试
encoded = tokenizer_bpe.encode("Hello, Hugging Face!")
print(f"BPE Encoded: {encoded.tokens}")


print("\n=== 2. 训练 WordPiece Tokenizer (BERT 用的) ===")
# 初始化 WordPiece 模型
tokenizer_wp = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer_wp.pre_tokenizer = pre_tokenizers.Whitespace()

# Trainer
trainer_wp = trainers.WordPieceTrainer(
    vocab_size=100,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

tokenizer_wp.train(["train_data.txt"], trainer_wp)
encoded = tokenizer_wp.encode("Hello, Hugging Face!")
print(f"WordPiece Encoded: {encoded.tokens}")


print("\n=== 3. 训练 Unigram Tokenizer (T5/SentencePiece 用的) ===")
# 初始化 Unigram 模型
tokenizer_uni = Tokenizer(Unigram())
tokenizer_uni.pre_tokenizer = pre_tokenizers.Whitespace()

# Trainer
trainer_uni = trainers.UnigramTrainer(
    vocab_size=100,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    unk_token="[UNK]"
)

tokenizer_uni.train(["train_data.txt"], trainer_uni)
encoded = tokenizer_uni.encode("Hello, Hugging Face!")
print(f"Unigram Encoded: {encoded.tokens}")

print("\n=== 总结 ===")
print("Hugging Face Tokenizers 支持的三大核心算法：")
print("1. BPE (Byte-Pair Encoding): 统计频次合并字符对。GPT 系列首选。")
print("2. WordPiece: 类似 BPE 但基于概率最大化选择合并。BERT 系列首选。")
print("3. Unigram: 从大词表里逐步删掉概率贡献小的词。T5/ALBERT 首选。")
