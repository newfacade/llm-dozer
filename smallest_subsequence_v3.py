def smallest_subsequence_v3(s: str) -> str:
    """
    Final corrected logic:
    We use the monotonic stack property to remove characters that are:
    1. Larger than the next character (stack[-1] > char)
    2. Redundant (appear later)
    
    AND we also handle the case where characters are EQUAL (stack[-1] == char).
    - If stack[-1] == char and stack[-1] is redundant, we SHOULD remove it to shorten the string.
      e.g., "bbac" -> remove first 'b' -> "bac". "bac" < "bbac".
    """
    from collections import Counter
    
    remaining = Counter(s)
    stack = []
    
    for char in s:
        # Check if stack top is >= current char AND appears later
        # If stack[-1] > char: remove stack[-1] to make lexicographically smaller
        # If stack[-1] == char: remove stack[-1] to make shorter (which is smaller)
        while stack and stack[-1] >= char and remaining[stack[-1]] > 0:
            stack.pop()
            # Note: We don't decrement remaining here because we're just peeking at future occurrences
        
        stack.append(char)
        remaining[char] -= 1
        
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
        print(f"Input: {t}, Output: {smallest_subsequence_v3(t)}")
