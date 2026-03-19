def smallest_subsequence(s: str) -> str:
    """
    返回可以通过删除重复字符形成的字典序最小的结果字符串。
    核心思路：单调栈 + 贪心 + 剩余可用字符检查
    """
    # 1. 记录每个字符最后出现的索引位置
    # 这让我们知道当我们想要从栈中弹出一个字符时，后面是否还有机会再次选到它
    last_occurrence = {char: i for i, char in enumerate(s)}
    
    stack = []
    # 使用集合来快速检查字符是否已经在栈中
    # 因为我们需要结果包含每个唯一字符恰好一次
    in_stack = set()
    
    for i, char in enumerate(s):
        # 如果字符已经在栈中，说明我们之前已经保留了该字符的一个实例
        # 并且那个实例的位置比当前位置更优（或者为了维持更小的字典序而被保留）
        if char in in_stack:
            continue
            
        # 核心逻辑：
        # 当栈不为空，且栈顶字符比当前字符大（stack[-1] > char）
        # 并且栈顶字符在后面还会出现（last_occurrence[stack[-1]] > i）
        # 我们就可以安全地弹出栈顶字符，因为它在后面还能被加回来，
        # 而现在用更小的当前字符替换它，能让整体字典序变小。
        while stack and stack[-1] > char and last_occurrence[stack[-1]] > i:
            removed_char = stack.pop()
            in_stack.remove(removed_char)
            
        stack.append(char)
        in_stack.add(char)
        
    return "".join(stack)

# 简单的测试用例
if __name__ == "__main__":
    test_cases = [
        "bcabc",      # 期望: "abc"
        "cbacdcbc",   # 期望: "acdb"
        "edebbed"     # 期望: "bed" (Wait, 'e' is at 0,2,6. 'd' at 1,6. 'b' at 3,4.)
                      # Unique: b, d, e.
                      # i=0, 'e'. Stack ['e']
                      # i=1, 'd'. 'd' < 'e'. Last 'e' is 6. Pop 'e'. Stack ['d'].
                      # i=2, 'e'. Stack ['d', 'e']
                      # i=3, 'b'. 'b' < 'e'. Last 'e' is 6. Pop 'e'.
                      #           'b' < 'd'. Last 'd' is 6. Pop 'd'. Stack ['b']
                      # i=4, 'b'. Skip.
                      # i=5, 'e'. Stack ['b', 'e']
                      # i=6, 'd'. 'd' < 'e'. Last 'e' is 6? NO. Last 'e' is 6. i is 6.
                      # Wait, last 'e' is at 6. i is 6. 6 > 6 False.
                      # So cannot pop 'e'.
                      # Stack ['b', 'e', 'd']
                      # Result: "bed"
    ]
    for t in test_cases:
        print(f"Input: {t}, Output: {smallest_subsequence(t)}")
