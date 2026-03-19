def smallest_subsequence_v2(s: str) -> str:
    """
    返回可以通过删除重复字符形成的字典序最小的结果字符串。
    策略：
    1. 使用单调栈贪心策略：如果当前字符比栈顶小，且栈顶字符在后面还会出现，则弹出栈顶。
       这保证了前面的字符尽可能小。
    2. 不强制去重：允许字符重复入栈（只要它比后面的字符小）。
    3. 后处理：如果栈尾的字符是重复的（即栈中其他位置也存在），则直接删除。
       因为末尾的重复字符只会增加长度，而不会通过“压制”后面的字符来降低字典序。
    """
    from collections import Counter
    
    # 统计剩余可用字符数量
    remaining = Counter(s)
    
    # 统计栈中字符数量（用于判断是否重复）
    stack_counts = Counter()
    
    stack = []
    
    for char in s:
        # 单调栈逻辑：
        # 当栈不为空
        # 且 栈顶字符 > 当前字符 (说明当前字符更适合放在这里)
        # 且 栈顶字符在后面还会出现 (说明我们可以安全地丢弃现在的栈顶，以后再捡回来)
        while stack and stack[-1] > char and remaining[stack[-1]] > 0:
            removed = stack.pop()
            stack_counts[removed] -= 1
        
        stack.append(char)
        stack_counts[char] += 1
        
        # 处理完当前字符，剩余数量减一
        remaining[char] -= 1
        
    # 后处理：从尾部删除多余的重复字符
    # 比如 "acdbc"，最后一个 'c' 是重复的，且后面没有字符了，删掉它能让字典序变小 ("acdb" < "acdbc")
    while stack and stack_counts[stack[-1]] > 1:
        removed = stack.pop()
        stack_counts[removed] -= 1
        
    return "".join(stack)

if __name__ == "__main__":
    test_cases = [
        "bcabc",      # Distinct: "abc". Non-distinct: "abc"
        "cbacdcbc",   # Distinct: "acdb". Non-distinct: "acdb" (Wait, my manual trace said acdb)
        "bcbd",       # Distinct: "bcd". Non-distinct: "bcbd"
        "bab",        # "ab"
        "aa",         # "a"
        "bbac"        # "bac"
    ]
    for t in test_cases:
        print(f"Input: {t}, Output: {smallest_subsequence_v2(t)}")
