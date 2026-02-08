from bpe_tokenizer import BPETokenizer

def test_basic_english():
    print("\n=== Test 1: Basic English ===")
    tokenizer = BPETokenizer()
    text = "Hello world! This is a test for BPE tokenizer."
    # 词表稍微大一点，保证能学到一些单词
    tokenizer.train(text, vocab_size=300)
    
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {text}")
    print(f"Encoded IDs: {encoded}")
    print(f"Decoded: {decoded}")
    
    assert text == decoded, "Basic English decoding failed!"
    print("PASS")

def test_chinese():
    print("\n=== Test 2: Chinese Handling ===")
    tokenizer = BPETokenizer()
    text = "你好世界！这是一个中文测试。"
    # 中文 UTF-8 字节很多，vocab_size 给大一点
    tokenizer.train(text, vocab_size=300)
    
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {text}")
    print(f"Encoded IDs: {encoded}")
    print(f"Decoded: {decoded}")
    
    assert text == decoded, "Chinese decoding failed!"
    print("PASS")

def test_special_tokens():
    print("\n=== Test 3: Special Tokens ===")
    tokenizer = BPETokenizer()
    text = "Hello <UNK> world <EOS>"
    special_tokens = ["<UNK>", "<EOS>"]
    
    # 注意：text 里包含了 special tokens，我们需要确保它们不被切碎
    tokenizer.train(text, vocab_size=300, special_tokens=special_tokens)
    
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {text}")
    print(f"Encoded IDs: {encoded}")
    
    # 验证 ID 是否正确 (Special tokens 应该在最后)
    unk_id = tokenizer.special_tokens["<UNK>"]
    eos_id = tokenizer.special_tokens["<EOS>"]
    print(f"Expected <UNK> ID: {unk_id}, <EOS> ID: {eos_id}")
    
    assert unk_id in encoded, "<UNK> ID not found in encoded list"
    assert eos_id in encoded, "<EOS> ID not found in encoded list"
    
    print(f"Decoded: {decoded}")
    assert text == decoded, "Special tokens decoding failed!"
    print("PASS")

def test_complex_merge():
    print("\n=== Test 4: Complex Merge Logic ===")
    tokenizer = BPETokenizer()
    # 构造一个高频重复的模式，强迫 BPE 进行多轮合并
    # aa -> A, bb -> B, AB -> C ...
    text = "aa " * 10 + "bb " * 10 + "aabb " * 10
    
    tokenizer.train(text, vocab_size=270) # 只要几个 merge 就够了
    
    encoded = tokenizer.encode("aabb")
    decoded = tokenizer.decode(encoded)
    
    print(f"Encoded 'aabb': {encoded}")
    print(f"Decoded: {decoded}")
    assert decoded == "aabb", "Complex merge decoding failed!"
    
    # 检查是否真的进行了压缩（ID 数量应该小于字节数 4）
    # aabb -> 4 bytes. 如果 merge 生效，应该小于 4 个 ID
    print(f"Length of encoded IDs: {len(encoded)} (Should be < 4)")
    assert len(encoded) < 4, "BPE compression failed!"
    print("PASS")

def test_save_load():
    print("\n=== Test 5: Save and Load ===")
    tokenizer = BPETokenizer()
    text = "Hello world! This is a test."
    tokenizer.train(text, vocab_size=280, special_tokens=["<UNK>"])
    
    # Save
    save_path = "tokenizer.json"
    tokenizer.save(save_path)
    print(f"Saved tokenizer to {save_path}")
    
    # Load into new tokenizer
    new_tokenizer = BPETokenizer()
    new_tokenizer.load(save_path)
    print("Loaded tokenizer")
    
    # Verify consistency
    encoded = tokenizer.encode(text)
    new_encoded = new_tokenizer.encode(text)
    
    print(f"Original Encoded: {encoded}")
    print(f"Loaded Encoded:   {new_encoded}")
    
    assert encoded == new_encoded, "Save/Load consistency check failed!"
    
    # Verify vocab size
    print(f"Original Vocab Size: {len(tokenizer.vocab)}")
    print(f"Loaded Vocab Size:   {len(new_tokenizer.vocab)}")
    assert len(tokenizer.vocab) == len(new_tokenizer.vocab), "Vocab size mismatch!"
    
    # Verify special tokens
    assert new_tokenizer.special_tokens["<UNK>"] == tokenizer.special_tokens["<UNK>"], "Special token mismatch!"
    
    # Cleanup
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
    print("PASS")

def test_tokenize_view():
    print("\n=== Test 6: Tokenize View ===")
    tokenizer = BPETokenizer()
    text = "Hello world! 你好"
    # 训练一个小词表，让 "Hello" 可能被合并，但 "你好" 可能会被打碎
    tokenizer.train(text, vocab_size=270)
    
    tokens = tokenizer.tokenize(text)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    
    # 验证返回的是列表
    assert isinstance(tokens, list), "Tokenize should return a list"
    # 验证拼接回来大致是对的（虽然 tokenize 里的 repr 会导致不能直接 join 回去）
    # 这里主要人工观察
    print("PASS")

if __name__ == "__main__":
    test_basic_english()
    test_chinese()
    test_special_tokens()
    test_complex_merge()
    test_save_load()
    test_tokenize_view()
    print("\nAll tests passed!")
