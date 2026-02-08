import re
import json
from collections import Counter
from tqdm import tqdm

class BPETokenizer:
    def __init__(self):
        self.merges = {}  # (p0, p1) -> new_id
        self.vocab = {}   # id -> bytes
        self.special_tokens = {} # str -> id
        self.inverse_special_tokens = {} # id -> str
        # GPT-4 风格的正则表达式
        # '(?:[sdmt]|ll|ve|re) 匹配常见缩写
        # ?\w+ 匹配单词
        # ?\d+ 匹配数字
        # ?[^\s\w\d]+ 匹配符号
        # \s+(?!\S) 匹配尾部空格
        # \s+ 匹配其他空格
        self.pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""")

    def train(self, text, vocab_size, special_tokens=None):
        """
        训练 BPE 分词器
        :param text: 训练文本
        :param vocab_size: 目标词表大小
        :param special_tokens: 特殊 token 列表，如 ["<unk>", "<pad>"]
        """
        print(f"Training BPE Tokenizer with target vocab size: {vocab_size}...")
        
        # 1. 预处理 Special Tokens 数量
        if special_tokens is None:
            special_tokens = []
        num_special_tokens = len(special_tokens)
        assert vocab_size >= 256 + num_special_tokens, "Vocab size must be at least 256 + special_tokens"
        
        # 计算 BPE 合并的目标 ID 上限
        # 我们需要保留最后的 N 个 ID 给 Special Tokens
        bpe_vocab_limit = vocab_size - num_special_tokens
        
        # 2. 预分词（Pre-tokenize）
        # 将文本切分成单词块，防止跨单词合并，例如 "dog." 中的 "g" 和 "." 不应该被合并
        text_chunks = re.findall(self.pattern, text)
        
        # 3. 统计 Chunk 频率并转换为初始字节序列
        chunk_counts = Counter(text_chunks)
        
        # ids_chunks: { "chunk_str": [byte_id1, byte_id2, ...] }
        # 初始状态下，ID 就是 0-255 的字节值
        ids_chunks = {chunk: [b for b in chunk.encode('utf-8')] for chunk in chunk_counts}
        
        # 初始化基础词表 (0-255)
        for i in range(256):
            self.vocab[i] = bytes([i])
        
        # 下一个可用的 ID (从 256 开始)
        next_id = 256
        
        # 4. 迭代合并 (Training Loop)
        num_merges = bpe_vocab_limit - 256
        with tqdm(total=num_merges, desc="Training BPE") as pbar:
            while len(self.vocab) < bpe_vocab_limit:
                # 统计当前所有 adjacent pairs 的频率
                stats = Counter()
                for chunk, freq in chunk_counts.items():
                    ids = ids_chunks[chunk]
                    for i in range(len(ids) - 1):
                        pair = (ids[i], ids[i+1])
                        stats[pair] += freq
                
                if not stats:
                    print("No more pairs to merge. Stopping early.")
                    break
                    
                # 找到频率最高的 pair
                # most_common(1) 返回 [(pair, count)]，取 [0][0] 得到 pair
                best_pair = stats.most_common(1)[0][0]
                
                # 记录合并规则
                self.merges[best_pair] = next_id
                
                # 更新词表：新 token 的字节序列 = 左 token 字节 + 右 token 字节
                self.vocab[next_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
                
                # 在所有 chunks 中应用合并
                # 这是一个简单的 O(N) 实现，效率一般但逻辑清晰
                for chunk in ids_chunks:
                    ids = ids_chunks[chunk]
                    new_ids = []
                    i = 0
                    while i < len(ids):
                        # 如果发现当前位置匹配 best_pair，则合并
                        if i < len(ids) - 1 and ids[i] == best_pair[0] and ids[i+1] == best_pair[1]:
                            new_ids.append(next_id)
                            i += 2
                        else:
                            new_ids.append(ids[i])
                            i += 1
                    ids_chunks[chunk] = new_ids
                
                next_id += 1
                pbar.update(1)
        
        # 5. 处理 Special Tokens (分配最后的 ID)
        if special_tokens:
            for token in special_tokens:
                # 确保 special tokens 不会覆盖已有的 ID
                # 注意：这里我们简单地将它们作为独立的 entry 加入词表
                # 它们没有对应的 merges 规则，因为它们是不可分割的整体
                self.vocab[next_id] = token.encode('utf-8')
                self.special_tokens[token] = next_id
                self.inverse_special_tokens[next_id] = token
                next_id += 1
                
        print(f"Training complete. Final vocab size: {len(self.vocab)}")
        print(f"Special tokens map: {self.special_tokens}")

    def encode(self, text):
        """
        将文本编码为 token ids
        """
        # 1. 处理 Special Tokens
        # 如果有 special tokens，我们需要先将它们从文本中切分出来，防止被 BPE 打碎
        if not self.special_tokens:
            special_pattern = None
        else:
            # 构造一个匹配任意 special token 的正则，注意要转义，排序是为了优先匹配更长的 token
            sorted_specials = sorted(self.special_tokens.keys(), key=len, reverse=True)
            special_pattern = re.compile("|".join(re.escape(k) for k in sorted_specials))

        # 最终的 ids 列表
        ids = []
        
        # 辅助函数：对一段没有 special token 的纯文本进行 BPE 编码
        def _encode_chunk(text_chunk):
            if not text_chunk:
                return []
            
            # 1. 预分词 (Regex split)
            words = re.findall(self.pattern, text_chunk)
            chunk_ids = []
            
            for word in words:
                # 转为字节序列
                word_bytes = [b for b in word.encode('utf-8')]
                
                # 2. BPE Merge
                # 这里我们需要不断合并，直到无法合并为止
                # 这是一个简单的实现：每次扫描所有可能的 pairs，找到在 merges 中最早出现的那个进行合并
                while len(word_bytes) >= 2:
                    # 找出当前序列中所有相邻的 pair
                    stats = {}
                    for i in range(len(word_bytes) - 1):
                        pair = (word_bytes[i], word_bytes[i+1])
                        # 检查这个 pair 是否在我们的合并规则中
                        if pair in self.merges:
                            stats[pair] = self.merges[pair] # 记录 pair -> new_id
                    
                    if not stats:
                        break # 没有可以合并的 pair 了
                    
                    # 找到优先级最高（new_id 最小，即最早被 merge）的 pair
                    # BPE 的合并顺序必须严格遵循训练时的顺序
                    best_pair = min(stats, key=lambda p: self.merges[p])
                    new_id = self.merges[best_pair]
                    
                    # 执行合并
                    new_word_bytes = []
                    i = 0
                    while i < len(word_bytes):
                        if i < len(word_bytes) - 1 and word_bytes[i] == best_pair[0] and word_bytes[i+1] == best_pair[1]:
                            new_word_bytes.append(new_id)
                            i += 2
                        else:
                            new_word_bytes.append(word_bytes[i])
                            i += 1
                    word_bytes = new_word_bytes
                
                chunk_ids.extend(word_bytes)
            return chunk_ids

        # 如果没有 special tokens，直接处理
        if not special_pattern:
            return _encode_chunk(text)

        # 如果有 special tokens，我们需要切分
        start = 0
        for match in special_pattern.finditer(text):
            # 处理前面的普通文本
            non_special_text = text[start:match.start()]
            if non_special_text:
                ids.extend(_encode_chunk(non_special_text))
            
            # 处理 special token
            special_token = match.group()
            ids.append(self.special_tokens[special_token])
            
            start = match.end()
        
        # 处理剩余的文本
        remaining_text = text[start:]
        if remaining_text:
            ids.extend(_encode_chunk(remaining_text))
            
        return ids

    def decode(self, ids):
        """
        将 token ids 解码为文本
        """
        text_parts = []
        current_bytes = []
        
        for idx in ids:
            # 如果是 Special Token
            if idx in self.inverse_special_tokens:
                # 先把积攒的 bytes 解码并加入
                if current_bytes:
                    text_parts.append(b"".join(current_bytes).decode('utf-8', errors='replace'))
                    current_bytes = []
                # 加入 special token 字符串
                text_parts.append(self.inverse_special_tokens[idx])
            else:
                # 如果是普通 Token，查表得到 bytes
                # 注意：self.vocab[idx] 可能是单个字节，也可能是合并后的字节序列
                if idx in self.vocab:
                    current_bytes.append(self.vocab[idx])
                else:
                    # 未知 token (理论上不应该发生，除非 vocab 没对齐)
                    pass
        
        # 处理最后剩余的 bytes
        if current_bytes:
            text_parts.append(b"".join(current_bytes).decode('utf-8', errors='replace'))
            
        return "".join(text_parts)

    def tokenize(self, text):
        """
        将文本切分为 token 字符串列表，便于观察分词结果
        对于无法解码为有效 UTF-8 的字节序列（如被切断的中文字符），将显示其字节表示（如 b'\\xe4'）
        """
        ids = self.encode(text)
        tokens = []
        for idx in ids:
            if idx in self.inverse_special_tokens:
                tokens.append(self.inverse_special_tokens[idx])
            elif idx in self.vocab:
                token_bytes = self.vocab[idx]
                try:
                    # 尝试解码为字符串
                    tokens.append(token_bytes.decode('utf-8'))
                except UnicodeDecodeError:
                    # 如果是无效的 utf-8 序列（比如被切断的多字节字符），显示其字节表示
                    tokens.append(str(token_bytes))
            else:
                # Fallback for unknown ids
                tokens.append(f"<ID:{idx}>")
        return tokens

    def save(self, file_path):
        """
        保存模型到 JSON 文件
        """
        # merges 的 key 是 tuple，JSON 不支持，转成 list 存储
        # 格式: [ [p0, p1], new_id ]
        merges_list = [[list(pair), new_id] for pair, new_id in self.merges.items()]
        
        model_data = {
            "merges": merges_list,
            "special_tokens": self.special_tokens
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)

    def load(self, file_path):
        """
        从 JSON 文件加载模型
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
            
        # 1. 恢复 merges
        # JSON 里的 list 变成了 [ [p0, p1], new_id ]
        self.merges = {tuple(pair): new_id for pair, new_id in model_data["merges"]}
        
        # 2. 恢复 special_tokens
        self.special_tokens = model_data["special_tokens"]
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        
        # 3. 重建 vocab
        self.vocab = {}
        # 3.1 基础字符 (0-255)
        for i in range(256):
            self.vocab[i] = bytes([i])
            
        # 3.2 根据 merges 重建组合 token
        # 必须按 new_id 从小到大顺序执行，因为后面的 token 可能依赖前面的
        sorted_merges = sorted(self.merges.items(), key=lambda item: item[1])
        for (p0, p1), new_id in sorted_merges:
            self.vocab[new_id] = self.vocab[p0] + self.vocab[p1]
            
        # 3.3 恢复 special tokens 的 bytes
        for token, idx in self.special_tokens.items():
            self.vocab[idx] = token.encode('utf-8')

