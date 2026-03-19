def smallestString(s: str) -> str: 
     """ 
     返回通过删除重复字符能形成的字典序最小字符串 
     
     规则：只能删除出现至少2次的字符中的某次出现，每个字符至少保留1次 
     """ 
     from collections import Counter 
     
     # 1. 统计每个字符的总出现次数 
     count = Counter(s) 
     # 注意：这里 count 实际上是 remaining_count
     # 在遍历之前，它代表总数。遍历过程中，它代表“后面还有多少个”。
     
     # 2. 使用栈构建结果（允许字符多次入栈） 
     stack = [] 
     
     # 3. 遍历字符串 
     for char in s: 
         # 每处理一个字符，剩余计数减1 
         count[char] -= 1 
         
         # 贪心：如果栈顶字符比当前字符大，且后面还会出现，则弹出 
         # 注意：这里不再检查 char 是否在栈中！ 
         while stack and stack[-1] > char and count[stack[-1]] > 0: 
             stack.pop() 
         
         # 将当前字符加入结果（即使它已经在栈中） 
         stack.append(char) 
     
     return ''.join(stack)

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
        print(f"Input: {t}, Output: {smallestString(t)}")
