### 1. 两数之和

* 解题

1. 双循环（巨慢）

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)-1):
            for j in range(i+1,len(nums)):
                if nums[i] + nums[j] == target:
                    return i,j
```

2. 哈希表

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = dict()
        for index,i in enumerate(nums):
            if (target-i) in dic:
                return (dic[target-i], index)
            dic[i] = index
        return (-1,-1)
```





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



### 4. 寻找两个有序数组的中位数

给定两个大小为 m 和 n 的有序数组 nums1 和 nums2。

请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。

你可以假设 nums1 和 nums2 不会同时为空。

```
nums1 = [1, 3]
nums2 = [2]

则中位数是 2.0

nums1 = [1, 2]
nums2 = [3, 4]

则中位数是 (2 + 3)/2 = 2.5
```

* 解题：归并数组之后再找位置

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        nums = []
        while nums1 and nums2:
            if nums1[0] < nums2[0]:
                nums.append(nums1.pop(0))
            else:
                nums.append(nums2.pop(0))
        if nums1:
            nums += nums1
        if nums2:
            nums += nums2
        n = len(nums)//2
        if len(nums)%2 == 1:
            return float(nums[n])
        if len(nums)%2 == 0:
            return (nums[n]+nums[n-1])/2
```



### 5. 最长回文子串（返回字串内容）

1. 动态规划

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        if n <= 1:
            return s
        maxlen = 0
        maxstr = s[0]
        dp = [[False for _ in range(n)] for _ in range(n)]
        for r in range(1,n):
            for l in range(r):
                if s[r] == s[l] and (r-l<=2 or dp[l+1][r-1]):
                    dp[l][r] = True
                    cur_len = r-l+1
                    if cur_len > maxlen:
                        maxlen = cur_len
                        maxstr = s[l:r+1]
        return maxstr
```

2. 找规律法：增加1个字符，回文串只能增加1个或2个，分别讨论即可

```python
class Solution:
    def longestPalindrome(self,s: str) -> str:
        if s == s[::-1]:
            return s
        start = 0
        maxlen = 0
        for i in range(len(s)):
            str1 = s[i - maxlen: i + 1]
            str2 = s[i - maxlen - 1: i + 1]
            if (i-maxlen)>=0 and (str1 == str1[::-1]):
                start,maxlen = i-maxlen,len(str1)
            if (i-maxlen)>= 1 and (str2== str2[::-1]):
                start,maxlen = i-maxlen-1,len(str2)
        return s[start:start+maxlen]
```



### 15. 三数之和

给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。

注意：答案中不可以包含重复的三元组。

```
例如, 给定数组 nums = [-1, 0, 1, 2, -1, -4]，

满足要求的三元组集合为：
[
  [-1, 0, 1],
  [-1, -1, 2]
]
```

* 解题：

1. 排序+哈希表+2sum(超时)

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        triplets = []
        n = len(nums)
        nums.sort()
        for i in range(n):
            target = 0-nums[i]
            dic = dict()
            for j in range(i+1,n):
                item_j = nums[j]
                if (target-item_j) in dic:
                    triplet = [nums[i], target-item_j, item_j]
                    if triplet not in triplets:
                        triplets.append(triplet)
                else:
                    dic[item_j] = j
        return triplets
```

2. 排序+固定指针+双指针

解题思路：

* 暴力法搜索为` O(N^3)O(N3)` 时间复杂度，可通过双指针动态消去无效解来优化效率。
* 双指针法铺垫： 先将给定 nums 排序，复杂度为 O(NlogN)O(NlogN)。
* 双指针法思路： 固定 33 个指针中最左（最小）数字的指针 k，双指针 i，j 分设在数组索引 (k, len(nums))(k,len(nums)) 两端，通过双指针交替向中间移动，记录对于每个固定指针 k 的所有满足 nums[k] + nums[i] + nums[j] == 0 的 i,j 组合：
  * 当 nums[k] > 0 时直接break跳出：因为 nums[j] >= nums[i] >= nums[k] > 0，即 33 个数字都大于 00 ，在此固定指针 k 之后不可能再找到结果了。
  * 当 k > 0且nums[k] == nums[k - 1]时即跳过此元素nums[k]：因为已经将 nums[k - 1] 的所有组合加入到结果中，本次双指针搜索只会得到重复组合。
  * i，j 分设在数组索引 (k, len(nums))(k,len(nums)) 两端，当i < j时循环计算s = nums[k] + nums[i] + nums[j]，并按照以下规则执行双指针移动：
    * 当s < 0时，i += 1并跳过所有重复的nums[i]；
    * 当s > 0时，j -= 1并跳过所有重复的nums[j]；
    * 当s == 0时，记录组合[k, i, j]至res，执行i += 1和j -= 1并跳过所有重复的nums[i]和nums[j]，防止记录到重复组合。
* 复杂度分析：
  * 时间复杂度 O(N^2)*O*(*N*2)：其中固定指针`k`循环复杂度 O(N)*O*(*N*)，双指针 `i`，`j` 复杂度 O(N)*O*(*N*)。
  * 空间复杂度 O(1)*O*(1)：指针使用常数大小的额外空间。

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res,k = [],0
        for k in range(len(nums)-2):
            if nums[k] > 0: break
            if k > 0 and nums[k] == nums[k-1]: continue
            i,j = k+1,len(nums)-1
            while i < j:
                s = nums[i]+nums[j]+nums[k]
                if s < 0:
                    i += 1
                    while i < j and nums[i] == nums[i-1]: i += 1
                elif s > 0:
                    j -= 1
                    while i < j and nums[j] == nums[j+1]: j -= 1
                else:
                    res.append([nums[k],nums[i],nums[j]])
                    i += 1
                    j -= 1
                    while i < j and nums[i] == nums[i-1]: i += 1
                    while i < j and nums[j] == nums[j+1]: j -= 1
        return res
```



### 16. 最接近的三数之和

给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。

```
例如，给定数组 nums = [-1，2，1，-4], 和 target = 1.
与 target 最接近的三个数的和为 2. (-1 + 2 + 1 = 2).
```

* 解题：

15题的扩展版：排序+固定一个指针+双指针

```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        n = len(nums)
        if n < 3: 
            return []
        res = float("inf")
        for i in range(n-2):
            if i > 0 and nums[i] == nums[i-1]:continue
            left = i+1
            right = n-1
            while left < right:
                cur = nums[i] + nums[left] + nums[right]
                if cur == target:
                    return target
                if abs(res-target) > abs(cur-target):
                    res = cur
                if cur < target:
                    left += 1
                elif cur > target:
                    right -= 1
        return res
```



### 26. 删除排序数组中的重复项

给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。

不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。

* 示例

```
nums = [1,1,2]
2
```

* 解题

1. 设置一个指针k来记录未重复数字的位置，前k个位置用于放置不同的数字

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return len(nums)
        k = 0
        for i in range(1,len(nums)):
            if nums[i] != nums[k]:
                k +=1
                nums[k] = nums[i]
        return k+1
```

2. Pythonic

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        for i in range(len(nums)-2,-1,-1):
            if nums[i] == nums[i+1]:
                del nums[i]
        return len(nums)
```



### 38. 报数

```python
class Solution:
    def countAndSay(self, n: int) -> str:
        prev_person = "1"
        for i in range(1,n):
            next_person,num,count = '',prev_person[0],1
            for j in range(1,len(prev_person)):
                if prev_person[j] == num:count+=1
                else:
                    next_person += str(count)+num
                    num = prev_person[j]
                    count = 1
            next_person += str(count) + num
            prev_person = next_person
        return prev_person
```



### 44. 通配符匹配

1. 双指针法

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        i = 0
        j = 0
        start = -1
        match = 0
        while i <len(s):
            ## 一对一匹配，匹配成功一起移动
            if j < len(p) and (s[i]==p[j] or p[j]=="?"):
                i +=1 
                j +=1
            ## 记录p的“*”的位置，还有s的位置
            elif j < len(p) and p[j] == "*":
                start = j
                match = i
                j += 1
            ## j回到记录的下一个位置，match更新下一个位置，这不代表用*匹配一个字符
            elif start != -1:
                j = start + 1
                match +=1
                i = match
            else:
                return False
        # 将多余的*直接匹配空串
        return all(x=="*" for x in p[j:])
```

2. 动态规划

`dp[i][j]`表示s的前i个和p的前j个能否匹配。

`dp[0][0]`什么都没有，所以为`True`

第一行`dp[0][j]`，换句话说s为空，与p匹配，所以只要p开始为*才为`True`

第一列`dp[i][0]`,当然全部为`False`

`dp[i][j-1]`表示，*代表是空字符

`dp[i-1][j]`表示，*代表是非空字符

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        sn = len(s)
        pn = len(p)
        dp = [[False]*(pn+1) for _ in range(sn+1)]
        dp[0][0] = True
        for j in range(1, pn+1):
            if p[j-1] == "*":
                dp[0][j] = dp[0][j-1]
        for i in range(1,sn+1):
            for j in range(1,pn+1):
                if (s[i-1] == p[j-1] or p[j-1] =="?"):
                    dp[i][j] = dp[i-1][j-1]
                elif p[j] == "*":
                    dp[i][j] = dp[i-1][j] or dp[i][j-1]
        return dp[-1][-1]
```



### 41. 缺失的第一个正数

给定一个未排序的整数数组，找出其中没有出现的最小的正整数。

```
示例 1:
输入: [1,2,0]
输出: 3
示例 2:

输入: [3,4,-1,1]
输出: 2
示例 3:

输入: [7,8,9,11,12]
输出: 1
```

* 解题

1. 查找法

```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        k = 1
        while k in nums:
            k += 1
        return k
```



### 58. 最后一个单词的长度

1. 

```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        s = s.split()
        return len(s[-1]) if s else 0
```

2. 

```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        if not s:return 0
        cnt = 0
        for i in reversed(s):
            if i == " ":
                if cnt != 0:
                    break
            else:
                cnt += 1
        return cnt
```

### 69. x的平方根

实现 int sqrt(int x) 函数。

计算并返回 x 的平方根，其中 x 是非负整数。

由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

```
输入: 4
输出: 2

输入: 8
输出: 2
说明: 8 的平方根是 2.82842..., 
     由于返回类型是整数，小数部分将被舍去。
```

* 解题：

1. 二分法

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        if x < 0:
            return -1
        elif x == 0:
            return 0
        l,r = 1,x
        while l +1 < r:
            mid = (l+r)//2
            if mid**2 == x:
                return mid
            elif mid**2 < x:
                l = mid
            else:
                r = mid
        return l
```

2. Pythonic

```
class Solution:
    def mySqrt(self, x: int) -> int:
        return int(x**0.5)
```



### 74. 搜索二维矩阵

编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：

* 每行中的整数从左到右按升序排列。
* 每行的第一个整数大于前一行的最后一个整数。

* 示例

```
输入:
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 3
输出: true
```

* 解题：

1. Pythonic(巨慢)

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        for i in matrix:
            if target in i:
                return True
        return False
```

2. 二分法（将二维矩阵转化为一维）

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix: return False
        row = len(matrix)
        col = len(matrix[0])
        left = 0
        right = row * col
        while left < right:
            mid = left + (right - left) // 2
            if matrix[mid // col][mid % col] < target:
                left = mid + 1
            else:
                right = mid
        return left < row * col and matrix[left // col][left % col] == target
```





### 80.删除排序数组中的重复项2

给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素最多出现两次，返回移除后数组的新长度。

不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。

```
给定 nums = [1,1,1,2,2,3],

函数应返回新长度 length = 5, 并且原数组的前五个元素被修改为 1, 1, 2, 2, 3 。

你不需要考虑数组中超出新长度后面的元素。
```

* 解题：

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        for i in range(len(nums)-3,-1,-1):
            if nums[i] == nums[i+1] == nums[i+2]:
                del nums[i]
        return len(nums)
```

### 88. 合并两个有序数组

1. 合并之后再排序

```python
def merge(self,nums1,m,nums2,n):
    nums1[:] = sorted(nums1[:m] + nums2)
```

2. 双指针（从后往前）

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        p1 = m -1
        p2 = n - 1
        p = m+n -1
        while p1 >= 0 and p2 >= 0:
            if nums1[p1] < nums2[p2]:
                nums1[p] = nums2[p2]
                p2 -= 1
            else:
                nums1[p] = nums1[p1]
                p1 -= 1
            p -= 1
        nums1[:p2+1] = nums2[:p2+1]
```

### 94. 二叉树的中序遍历

* 解题

1. 颜色标注法

**使用颜色标记节点的状态，新节点为白色，已访问的节点为灰色。**
**如果遇到的节点为白色，则将其标记为灰色，然后将其右子节点、自身、左子节点依次入栈。**
**如果遇到的节点为灰色，则将节点的值输出。**

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        WHITE, GRAY = 0, 1
        res = []
        stack = [(WHITE, root)]
        while stack:
            color, node = stack.pop()
            if node is None: continue
            if color == WHITE:
                stack.append((WHITE, node.right))
                stack.append((GRAY, node))
                stack.append((WHITE, node.left))
            else:
                res.append(node.val)
        return res
```

2. 分治法

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None:
            return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
```



### 96. 不同的二叉搜索树

给定一个整数 *n*，求以 1 ... *n* 为节点组成的二叉搜索树有多少种？

```
输入：3
输出：5
```

* 解题：

BST的个数只与有序数列的大小有关，而与具体的值没关系（例如：123和456的BST个数相同）

```python
class Solution:
    def numTrees(self, n: int) -> int:
        if n < 0:
            return -1
        count = [0]*(n+1)
        count[0] = 1
        for i in range(1,n+1):
            for j in range(i):
                count[i] += count[j] * count[i-j-1]
        return count[n]
```



### 102. 二叉树的层次遍历

给定一个二叉树，返回其按层次遍历的节点值。 （即逐层地，从左到右访问所有节点）。

例如:
给定二叉树: `[3,9,20,null,null,15,7]`

```
[
  [3],
  [9,20],
  [15,7]
]
```

* 解题

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        levels = []
        if not root:return levels
        def helper(node,level):
            if len(levels) == level:
                levels.append([])
            levels[level].append(node.val)
            if node.left:
                helper(node.left,level+1)
            if node.right:
                helper(node.right,level+1)
        helper(root,0)
        return levels
```

### 103. 二叉树的最大深度

* 解题

1. 迭代

```python
class Solution:
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """ 
        stack = []
        if root is not None:
            stack.append((1, root))
        
        depth = 0
        while stack != []:
            current_depth, root = stack.pop()
            if root is not None:
                depth = max(depth, current_depth)
                stack.append((current_depth + 1, root.left))
                stack.append((current_depth + 1, root.right))
        
        return depth
```

2. 递归

```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:return 0
        left_heights = self.maxDepth(root.left)
        right_heights = self.maxDepth(root.right)
        return max(left_heights,right_heights)+1
```



### 107. 二叉树的层次遍历2

给定一个二叉树，返回其节点值自底向上的层次遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）

给定二叉树 `[3,9,20,null,null,15,7]`

```
   3
   / \
  9  20
    /  \
   15   7
```

返回其自底向上的层次遍历为：

```
[
  [15,7],
  [9,20],
  [3]
]
```

* 解题

```python
class Solution:
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        levels = []
        if not root:return levels
        def helper(node,level):
            if len(levels) == level:
                levels.append([])
            levels[level].append(node.val)
            if node.left:
                helper(node.left,level+1)
            if node.right:
                helper(node.right,level+1)
        helper(root,0)
        return levels[::-1]
```



### 136. 只出现一次的数字

给定一个**非空**整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

```
输入: [2,2,1]
输出: 1

输入: [4,1,2,1,2]
输出: 4
```

* Solution:

1. 

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        dic = dict()
        for num in nums:
            if num not in dic:
                dic[num] = 1
            else:
                del dic[num]
        for i in dic.keys():
            return i
```

2. 

### 144. 二叉树的前序遍历

给定一个二叉树，返回它的 *前序* 遍历。

* 解题：

1. 分治法：递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if root == None:
            return []
        return [root.val] + self.preorderTraversal(root.left)+self.preorderTraversal(root.right)
```

2. 迭代：用栈来保存每个树节点，再用一个列表来保存树的val

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None:
            return []
        res = []
        s = []
        s.append(root)
        while s:
            root = s.pop()
            res.append(root.val)
            if root.right is not None:
                s.append(root.right)
            if root.left is not None:
                s.append(root.left)
        return res
```

### 145. 二叉树的后序遍历

* 解题

1. 分治

```python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None:
            return []
        return self.postorderTraversal(root.left) + self.postorderTraversal(root.right)+[root.val]
```

2. 颜色标注法

```python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        WHITE,GREY = 0,1
        res = []
        stack = [(WHITE,root)]
        while stack:
            color,node = stack.pop()
            if node is None:continue
            if color == WHITE:
                stack.append((GREY,node))
                stack.append((WHITE,node.right))
                stack.append((WHITE,node.left))
            else:
                res.append(node.val)
        return res
```



### 162.* 寻找峰值

峰值元素是指其值大于左右相邻值的元素。

给定一个输入数组 nums，其中 nums[i] ≠ nums[i+1]，找到峰值元素并返回其索引。

数组可能包含多个峰值，在这种情况下，返回任何一个峰值所在位置即可。

你可以假设 nums[-1] = nums[n] = -∞。

```
输入: nums = [1,2,3,1]
输出: 2
解释: 3 是峰值元素，你的函数应该返回其索引 2。
```

1. 遍历

```python
class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums = [float("-inf")] + nums + [float("-inf")]
        n = len(nums)
        for i in range(1, n - 1):
            if nums[i - 1] < nums[i] and nums[i] > nums[i + 1]:
                return i - 1

```

2. 遍历

```python
class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        for i in range(1, len(nums)):
            if nums[i - 1] > nums[i]:
                return i - 1
        return len(nums) - 1

```

3. 二分法

```

```

### 172. 阶乘后的0

给定一个整数 n，返回 n! 结果尾数中零的数量。

```
示例 1:

输入: 3
输出: 0
解释: 3! = 6, 尾数中没有零。
示例 2:

输入: 5
输出: 1
解释: 5! = 120, 尾数中有 1 个零.
```

说明: 你算法的时间复杂度应为 O(log n) 。

* 解题：计算质因数位5的个数

```python
class Solution:
    def trailingZeroes(self, n: int) -> int:
        if n < 0:
            return -1
        count = 0
        while n > 0:
            n //= 5
            count += n
        return count
```

### 202. 快乐数

编写一个算法来判断一个数是不是“快乐数”。

一个“快乐数”定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数变为 1，也可能是无限循环但始终变不到 1。如果可以变为 1，那么这个数就是快乐数。

```
输入: 19
输出: true
解释: 
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1
```

* 解题：

如果一个数不是快乐数的话，那么必然会出现循环的，判断是不是1引起的循环。

1. 数学计算

```python
class Solution:
    def squresum(self,n):
        res = 0
        while n > 0:
            bit = n % 10
            res += bit*bit
            n = n//10
        return res
    def isHappy(self, n: int) -> bool:
        slow,fast = n,self.squresum(n)
        while slow != fast:
            slow = self.squresum(slow)
            fast = self.squresum(fast)
            fast = self.squresum(fast)
        return slow == 1
        
```

2. 字符串计算

```python
class Solution:
    def isHappy(self, n: int) -> bool:
        n = str(n)
        slow = n
        fast = str(sum(int(i) ** 2 for i in n))
        while slow != fast:
            slow = str(sum(int(i) ** 2 for i in slow))
            fast = str(sum(int(i) ** 2 for i in fast))
            fast = str(sum(int(i) ** 2 for i in fast))
        return slow == "1"
```

### 233.*  数字1的个数

给定一个整数 n，计算所有小于等于 n 的非负整数中数字 1 出现的个数。

```
输入: 13
输出: 6 
解释: 数字 1 出现在以下数字中: 1, 10, 11, 12, 13 。
```

* 解题：

Python采用字符串处理的方法会超时

```python
class Solution:
    def __init__(self):
        self.map = {0:0, 9:1}
        for i in range(1, 10):
            self.map[10**(i+1)-1] = 10**i+10*(self.map[10**i-1])
    
    def countDigitOne(self, n):
        if n<=0:
            return 0
        i = 1
        while i*10<=n:
            i *= 10
        return int(self.count(n ,i))

    def count(self, n, i):
        if n==0:
            return 0
        else:
            while i>n:
                i /= 10
            n_1 = self.map[i-1]
            return min(i, n-i+1)+n//i*n_1+self.count(n%i, i/10)
```



### 240. 搜索二维矩阵

编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target。该矩阵具有以下特性：

每行的元素从左到右升序排列。
每列的元素从上到下升序排列。

* 示例

```
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]

给定 target = 5，返回 true。

给定 target = 20，返回 false。
```



* 解题：从右上角开始扫描

```python
class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix or not matrix[0]:
            return False
        occ = False
        row,col = 0, len(matrix[0])-1
        while row < len(matrix) and col >=0:
            if matrix[row][col] == target:
                occ = True
                col -=1
            elif matrix[row][col] < target:
                row += 1
            else:
                col -= 1
        return occ
```



### 263. 丑数

编写一个程序判断给定的数是否为丑数。

丑数就是只包含质因数 `2, 3, 5` 的**正整数**。

* 解题：递归

```python
class Solution:
    def isUgly(self, num: int) -> bool:
        if num <= 0:
            return False
        for i in [2,3,5]:
            while num % i == 0:
                num //= i
        return num ==1
```



### g4g1. 求连续子数组的和为k

* 输入: 

   [1,4,20,3,10,5],sum =  33

* 输出：

  Sum found between indexes 2 and 4

* 如果未找到子数组，输出No subarray found

* 解题

```python
def subArraySum(arr,n,sum):
    curr_sum = arr[0]
    start = 0
    i = 1
    while i <= n:
        while curr_sum > sum and start < i-1:
            curr_sum -= arr[start]
            start += 1
        if curr_sum == sum:
            print("Sum found between indexes")
            print("%d and %d"%(start,i-1))
            return 1
        if i < n:
            curr_sum += arr[i]
        i += 1
    print("No subarray found")
    return 0
```

### lintcode6. Merge Sorted Array 2

Merge two given sorted ascending integer array *A* and *B* into a new sorted integer array.

### **Example**

**Example 1:**

```
Input:  A=[1], B=[1]
Output: [1,1]	
Explanation:  return array merged.
```

**Example 2:**

```
Input:  A=[1,2,3,4], B=[2,4,5,6]
Output: [1,2,2,3,4,4,5,6]	
Explanation: return array merged.
```

* 解题：

1. Pythonic

```python
class Solution:
    """
    @param A: sorted integer array A
    @param B: sorted integer array B
    @return: A new sorted integer array
    """
    def mergeSortedArray(self, A, B):
        # write your code here
        return sorted(A+B)

```

2. 常规双指针

```python
class Solution:
    """
    @param A: sorted integer array A
    @param B: sorted integer array B
    @return: A new sorted integer array
    """
    def mergeSortedArray(self, A, B):
        # write your code here
        res = []
        a,b = len(A),len(B)
        i,j = 0,0
        while i < a and j < b:
            if A[i] < B[j]:
                res.append(A[i])
                i += 1 
            else:
                res.append(B[j])
                j += 1 
        while i < a:
            res.append(A[i])
            i += 1 
        while j < b:
            res.append(B[j])
            j += 1 
        return res 
```

3. 升级版

```python
def merge(A,B):
    res = []
    while A and B:
        if A[0] < B[0]:
            res.append(A.pop(0))
        else:
            res.append(B.pop(0))
    if A:
        res += A
    if B:
        res += B
    return res
```



### lintcode50. Product of Array Exclude Itself

```
Given an integers array A.
Define B[i] = A[0] * ... * A[i-1] * A[i+1] * ... * A[n-1], calculate B WITHOUT divi
de operation.
Example
For A=[1, 2, 3], return [6, 3, 2]
```

解题：原地求积

设置两个循环

1. 记录数组从后往前的累乘结果，f[i]表示i位之后所有元素的乘积
2. 从左往右，跳过i左侧累乘，右侧直接乘以f[i+1]

```python
class Solution:
    """
    @param: nums: Given an integers array A
    @return: A long long array B and B[i]= A[0] * ... * A[i-1] * A[i+1] * ... * A[n-1]
    """
    def productExcludeItself(self, nums):
        # write your code here
        length, B = len(nums),[]
        f = [0 for _ in range(len(nums)+1)]
        f[length] = 1
        for i in range(length-1,0,-1):
            f[i] = f[i+1]*nums[i]
        tmp = 1 
        for i in range(length):
            B.append(tmp*f[i+1])
            tmp *= nums[i]
        return B
```

### lintcode 31. Partition Array

Given an array `nums` of integers and an int `k`, partition the array (i.e move the elements in "nums") such that:

- All elements < *k* are moved to the *left*
- All elements >= *k* are moved to the *right*

Return the partitioning index, i.e the first index *i* nums[*i*] >= *k*.

Example 1:

```
Input:
[],9
Output:
0
```

Example 2:

```
Input:
[3,2,2,1],2
Output:1
Explanation:
the real array is[1,2,2,3].So return 1
```

* 解题：

1. Pythonic：

```python
from bisect import bisect_left
def partitionArray(self,nums,k):
    nums.sort()
    return bisect_left(nums,k)
```

2. 双指针法

```python
class Solution:
    """
    @param nums: The integer array you should partition
    @param k: An integer
    @return: The index after partition
    """
    def partitionArray(self,nums,k):
        left = 0
        right = len(nums)-1
        while left <= right:
            if nums[left] < k:
                left += 1
            elif nums[right] >= k:
                right -= 1
            else:
                tmp = nums[left]
                nums[left] = nums[right]
                nums[right] = tmp
                left += 1
                right -= 1
        return left
```

### Lintcode 140. Fast Power

Calculate the **an % b** where a, b and n are all 32bit non-negative integers.

```
For 231 % 3 = 2

For 1001000 % 1000 = 0

```

时间复杂度：O(logn)

* 解题

递归，根据公式`(a*b)%p = ((a%p)*(b%p))%p`

```python
class Solution:
    """
    @param a: A 32bit integer
    @param b: A 32bit integer
    @param n: A 32bit integer
    @return: An integer
    """
    def fastPower(self, a, b, n):
        # write your code here
        if n == 1:
            return a % b
        elif n == 0:
            return 1 % b
        elif n < 0:
            return -1
        
        num = self.fastPower(a,b,n//2)
        num = (num*num)%b
        if n % 2 == 1:
            num = (num * a)%b
        return num
```



### lintcode 183. Wood Cut

Given n pieces of wood with length `L[i]` (integer array). Cut them into small pieces to guarantee you could have equal or more than k pieces with the same length. What is the longest length you can get from the n pieces of wood? Given L & k, return the maximum length of the small pieces.

### **Example**

**Example 1**

```plain
Input:
L = [232, 124, 456]
k = 7
Output: 114
Explanation: We can cut it into 7 pieces if any piece is 114cm long, however we can't cut it into 7 pieces if any piece is 115cm long.
```

**Example 2**

```plain
Input:
L = [1, 2, 3]
k = 7
Output: 0
Explanation: It is obvious we can't make it.
```

* Solution:

### lintcode 365. 统计二进制表示中1的个数

注意python中负数的统计

```python
class Solution:
    """
    @param: num: An integer
    @return: An integer
    """
    def countOnes(self, num):
        # write your code here
        count = 0
        flag = False
        if num < 0:
            num = abs(num)-1
            flag = True
        while num > 0:
            num = num&(num-1)
            count += 1 
        if flag:
            count = 32-count
        return count
```

2. 

### lintcode 373. Partition Array by Odd and Even

* Question

  ```
  Partition an integers array into odd number first and even number second.
  Example
  Given [1, 2, 3, 4], return [1, 3, 2, 4]
  Challenge
  Do it in-place. 
  ```

* 解题：

```python
class Solution:
    """
    @param: nums: an array of integers
    @return: nothing
    """
    def partitionArray(self, nums):
        # write your code here
        l = 0
        r = len(nums)-1 
        while l < r:
            if nums[l]%2 == 0 and nums[r]%2 == 1:
                nums[l],nums[r] = nums[r],nums[l]
                l += 1 
                r -= 1 
            elif nums[l] %2 == 1:
                l += 1 
            elif nums[r] %2 == 0:
                r -= 1 
        return nums
```

