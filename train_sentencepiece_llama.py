import sentencepiece as spm
import os

# 1. 准备训练数据
# 实际训练时这里应该是你的海量文本文件，每行一个句子
# 这里我们造一个简单的 demo 数据集
with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write("The quick brown fox jumps over the lazy dog.\n" * 100)
    f.write("你好，世界！这是一个测试句子。\n" * 100)
    f.write("Hugging Face is awesome.\n" * 100)

# 2. 训练 SentencePiece 模型
# 这是 Llama 1/2 原始使用的训练方式
# input: 输入文件路径
# model_prefix: 输出模型的前缀名
# vocab_size: 词表大小 (Llama 通常是 32000)
# character_coverage: 字符覆盖率 (1.0 表示覆盖所有字符，通常中文设 0.9995 过滤生僻字)
# model_type: 模型类型 (bpe, unigram, char, word)
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='llama_tokenizer_demo',
    vocab_size=100, # 数据量太小，调小词表大小以演示
    character_coverage=1.0,
    model_type='bpe',
    user_defined_symbols=['<pad>', '<s>', '</s>'], # 自定义特殊符号
    # 下面这些是优化参数
    input_sentence_size=1000000, # 用于训练的句子数量上限
    shuffle_input_sentence=True
)

# 3. 加载并测试
sp = spm.SentencePieceProcessor()
sp.load('llama_tokenizer_demo.model')

text = "The quick brown fox jumps over the lazy dog. 你好世界"
print(f"Original: {text}")

# Encode: 文本 -> ID
ids = sp.encode_as_ids(text)
print(f"IDs: {ids}")

# Decode: ID -> 文本
decoded = sp.decode_ids(ids)
print(f"Decoded: {decoded}")

# Encode pieces: 文本 -> 切分后的 token 字符串
pieces = sp.encode_as_pieces(text)
print(f"Pieces: {pieces}")

# 4. 转换为 Hugging Face 格式 (可选)
# 如果你想在 transformers 库里用，通常需要转换
# from transformers import LlamaTokenizer
# tokenizer = LlamaTokenizer(vocab_file="llama_tokenizer_demo.model")
# tokenizer.save_pretrained("./hf_llama_tokenizer")
