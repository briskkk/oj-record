### 1.

- 题目描述：给定两个字符集合，一个为全量字符集，一个为已占用字符集，已占用的字符集中的字符不能再使用，要求输出剩余可用字符集

- 输入描述：

输入为一个字符串，字符串中包含了全量字符集和已占用字符集，两个字符集使用@连接，@前的字符集合为全量字符集，@后的字符集合为已占用字符集。

已占用字符集中的字符一定是全量字符集中的字符

字符集中的字符和字符之间使用英文逗号分隔

字符集中的字符标识为字符加数字，字符跟数字使用英文冒号分隔，比如`a:1`,表示1个a字符，字符只考虑英文字母，区分大小写，数字只考虑正整型，数量不超过100，如果一个字符都没有被占用，@标识仍然存在，例如`a:3,b:5,c:2@`

- 输出描述

可用字符集，输出带回车换行

- 示例

```
输入：a:3,b:5,c:2@a:1,b:2
输出：a:2,b:3,c:2
```

输出的字符顺序要跟输入一致

- 解题：

```python
ql,zy = input().strip().split("@")
ql = ql.split(",")
zy = zy.split(",")
a = []
b = []
for i in range(len(ql)):
    a.append(ql[i][0])
    b.append(int(ql[i][2]))
dicc = dict(zip(a,b))
for i in range(len(zy)):
    dicc[zy[i][0]] -= int(zy[i][2])
out = []
for i,j in dicc.items():
    j = str(j)
    i += j
    out.append(":".join(i))
print(",".join(out))
```



### 2.

如下图是一棵Trie树，圆圈标识内部节点，指向孩子节点的没饿过标记的值范围在0-255之间，每个内部节点最多有256个还自己欸点，三角形标识叶子节点，每个叶子节点中存储一个value，根节点到叶子节点之间路径上的所有字符构成一个完整key.

Trie树可采用Post Order Unary Degree Sequence及逆行唯一编码，上图的另一种表达方式如下

labels：指向每个分支的标记，每个标记的值在0-255之间，实际在存储时使用1B的无符号证书存储字符对应的ASCII码。

Haschild：标记每个分支下的节点是否是内部节点

POUDS：标记兄弟节点边界，每组兄弟节点的第一个节点1标识，剩下的用0标识

Values：叶子节点内容

现在有一颗使用POUDS编码的Trie树，树中key长度均相同，输入要查找的key，输出key对应的value

- 输入描述

第一行的数字M表示labels、haschild、POUDS数据大小，紧跟着的3行分表表示Labels、HasChild、POUDS数组内容，用空格分开，第五行的数字N表示values数组大小，随后1行表示values数组内容，第7行的数字表示key数组大小，随后一行表示要查找的key字符数组。Labels数组的每个字符的取值范围在0-255之间，HasChild和POUDS数组的取值是0或1.

- 输出描述

输出一行key对应的value，若key不存在，输出0

- 示例

输入

```shell
15
115 112 116 97 111 121 114 101 105 112 121 114 102 115 116
0 0 0 1 1 0 1 0 0 0 0 1 1 1 1
```

输出：

```shell
7
```



### 3. 逻辑计算器

常用的逻辑计算有and(&),or(|),not(!)

它们的逻辑是：

```python
1&1 = 1
1&0 = 0
0&1 = 0
0&0 = 0
1|1 = 1
1|0 = 0
0|1 = 0
!0 = 1
!1 = 0
```

其中他们的优先级关系是：not>and>or

例如：

```
A|B&C实际是A|(B&C)
A&B|C&D实际是(A&B)|(C&D)
!A&B|C实际是((!A)&B)|C
```

示例：

```
输入：!(1&0)|0&1
输出：1
输入：!(1&0)&0|0
输出：
```

将字符串分为2累，一类是有括号的字符串，一类是不含括号的，对于含有括号的字符串，可以先将它转化为没有括号的字符串，问题转化为消解括号和计算没有括号的逻辑字符串的逻辑值。

1. 消解括号：可以通过一个栈来匹配消解括号，）总是和最近的（进行匹配，先将字符串不断送入栈中，碰到）的时候，在不断从栈中取出元素，计算两个括号之间字符串的逻辑值
2. 计算无括号的字符串逻辑值：安装优先级的顺序从高优先级到低优先级不断消减逻辑判断符号，最终得到逻辑计算结果

```python
def calnotkuohao(s):
    slen = len(s)
    out = ""
    cnt = 0
    while cnt < slen:
        if s[cnt] == "!":
            out = str(int(not int(s[cnt+1])))
            cnt += 2
        else:
            out += s[cnt]
            cnt += 1
    s = out
    slen = len(s)
    out = ""
    cnt = 0
    while cnt < slen:
        if s[cnt] == "&":
            a = int(out[-1])
            out = out[:-1]
            out += str(int(int(s[cnt+1]) and a))
            cnt += 2
        else:
            out += s[cnt]
            cnt += 1
    s = out
    slen = len(s)
    out = ""
    cnt = 0
    while cnt < slen:
        if s[cnt] == "|":
            a = int(out[-1])
            out = out[:-1]
            out += str(int(int(s[cnt +1]) or a))
            cnt += 2
        else:
            out += s[cnt]
            cnt += 1
    return out

in_string = input().strip()
stack1 = []
for i in range(len(in_string)):
    if in_string[i] == ")":
        stackup = stack1.pop()
        strtmp = ""
        while stackip != "(":
            strtmp = stackup + strtmp
            stackip = stack1.pop()
        stack1.append(calnotkuohao(strtmp))
    else:
        stack1.append(in_string[i])
last = "".join(stack1)
print(calnotkuohao(last))

```

