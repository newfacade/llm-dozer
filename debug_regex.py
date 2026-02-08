import re

# 1. 定义正则
# 注意：为了让正则更易读，我把 pattern 拆开写，效果是一样的
PAT_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+"""
pattern = re.compile(PAT_STR)

# 2. 定义测试文本
text = "That's 100% correct! I'll pay $50.  You're 2 good 4 me.   "

# 3. 执行切分
tokens = pattern.findall(text)

# 4. 详细分析每个 token 是由哪部分正则匹配的
def identify_match(token):
    # 按优先级顺序检查
    if re.fullmatch(r"'(?:[sdmt]|ll|ve|re)", token):
        return "1. 缩写 (Contract)"
    elif re.fullmatch(r" ?\w+", token):
        return "2. 单词 (Word)"
    elif re.fullmatch(r" ?\d+", token):
        return "3. 数字 (Number)"
    elif re.fullmatch(r" ?[^\s\w\d]+", token):
        return "4. 符号 (Punct)"
    elif re.fullmatch(r"\s+(?!\S)", token): # 注意：这在独立检查时可能很难完全复现 lookahead，但这里我们可以通过内容判断
        if token.strip() == "": 
            return "5. 尾部空格 (Trailing Space)"
        return "未知"
    elif re.fullmatch(r"\s+", token):
        return "6. 普通空格 (Space)"
    else:
        return "未知 (Unknown)"

print(f"原始文本: \"{text}\"")
print("-" * 60)
print(f"{'Token (显示)':<15} | {'Token (原始)':<15} | 匹配规则")
print("-" * 60)

for token in tokens:
    # 替换空格为 Ġ 以便观察
    display = token.replace(' ', 'Ġ')
    rule = identify_match(token)
    
    # 修正逻辑：identify_match 对于 lookahead 的判断可能不准，
    # 因为单独拿出来看 '   ' 既符合 \s+ 也符合 \s+(?!\S)。
    # 但根据 tokenizer 逻辑，只有最后的空格才会被归为 rule 5。
    if rule == "6. 普通空格 (Space)" and token == tokens[-1]:
         rule = "5. 尾部空格 (Trailing Space)"

    print(f"{display:<15} | {repr(token):<15} | {rule}")
