def smallest_subsequence_v4(s: str) -> str:
    """
    Final Logic:
    1. Monotonic Stack: Remove stack[-1] if stack[-1] >= char AND stack[-1] appears later.
    2. Post-processing: Remove redundant characters from the END of the stack.
       A character at the end is redundant if it appears earlier in the stack.
    """
    from collections import Counter
    
    remaining = Counter(s)
    stack = []
    
    # Pass 1: Monotonic Stack (Forward)
    for char in s:
        # If stack top is >= current char AND appears later, remove it.
        # This handles cases like "bab" -> "ab", "bbac" -> "bac".
        # But for "bcbd", 'b' < 'c', so 'b' stays. 'c' > 'b' but 'c' is unique, so 'c' stays.
        while stack and stack[-1] >= char and remaining[stack[-1]] > 0:
            stack.pop()
        
        stack.append(char)
        remaining[char] -= 1
        
    # Pass 2: Remove trailing redundant characters (Backward)
    # e.g., "acdbc" -> remove last 'c' -> "acdb".
    # We need to count occurrences IN THE STACK.
    stack_counts = Counter(stack)
    final_stack = []
    
    # We iterate backwards. If a char is redundant (count > 1), we remove it from the END.
    # Why? Because removing from the end shortens the string, which is lexicographically smaller.
    # "acdbc" vs "acdb". "acdb" is prefix of "acdbc", so "acdb" < "acdbc".
    
    # But wait, removing from the end is only good if the char is redundant.
    # If we remove unique char, we violate the problem statement (must keep at least one).
    
    # We can just pop from stack end while the top element count > 1.
    while stack and stack_counts[stack[-1]] > 1:
        removed = stack.pop()
        stack_counts[removed] -= 1
        
    return "".join(stack)

if __name__ == "__main__":
    test_cases = [
        "bcabc",      # "abc"
        "cbacdcbc",   # "acdb"
        "bcbd",       # "bcbd"
        "bab",        # "ab"
        "aa",         # "a"
        "bbac"        # "bac"
    ]
    for t in test_cases:
        print(f"Input: {t}, Output: {smallest_subsequence_v4(t)}")
