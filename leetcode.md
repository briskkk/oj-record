### 1. 实现字符串

1. 

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if needle in haystack:
            return haystack.index(needle)
        else:
            return -1
```

2. 

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        c = -1
        m = len(haystack)
        n = len(needle)
        for i in range(m-n+1):
            if haystack[i:i+n] == needle:
                c = i
                break
        return c
```



