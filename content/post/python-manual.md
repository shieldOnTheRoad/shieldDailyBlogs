---
title: "Python Manual"
date: 2018-02-06T12:48:36+08:00
lastmod: 2018-02-08T08:48:36+08:00
draft: false
tags: ["python", "code"]
categories: ["Program"]
author: 'shield'

weight: 10

# You can also close(false) or open(true) something for this content.
# P.S. comment can only be closed
# comment: false
# autoCollapseToc: false
# reward: false
toc: false
mathjax: true
---

我的`python`速查手册。计算机语言这玩意，经久不用就容易忘记，记录下来。

<!--more-->
<br>

## #1 **PYTHON API**
#### **1.1 Map,Filter和Reduce**
`Map`会将一个函数映射到一个输入列表的所有元素上。
```python
# case1: items传递普通变量（e.g., 数值）
items = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, items))
print(squared)

# case2: items传递特殊变量（e.g., 函数)
def multiply(x):
        return (x*x)
def add(x):
        return (x+x)

funcs = [multiply, add]
for i in range(5):
    value = map(lambda x: x(i), funcs)
    print(list(value))  # 转成list适应python3返回的迭代器
```
`filter`过滤列表中的元素，并且返回一个由所有符合要求（返回值为`True`）的元素所构成的列表。
```python
number_list = range(-5, 5)
less_than_zero = filter(lambda x: x < 0, number_list)
print(list(less_than_zero)) 

# Output: [-5, -4, -3, -2, -1]
```
`reduce`对一个列表进行一些计算并返回结果
```python
product = reduce((lambda x, y: x * y), [1, 2, 3, 4])
print(product)
```

<br>

## #2 **知识点**
#### **2.1 Python debugger(pdb)**
命令行使用python debugger运行一个脚本。
```shell
$ python -m pdb my_script.py
```
py脚本内嵌入pdb模块，使用`pdb.set_trace()`方法设置断点。
```python
import pdb

def make_bread():
    pdb.set_trace()
    return "I don't have time"

print(make_bread())

# c: 继续执行
# w: 显示当前正在执行的代码行的上下文信息
# a: 打印当前函数的参数列表
# s: 执行当前代码行，并停在第一个能停的地方（相当于单步进入）
# n: 继续执行到当前函数的下一行，或者当前行直接返回（单步跳过）
```

<br>

## # **参考资料**
1. [Python进阶](http://docs.pythontab.com/interpy/)
2. [Intermediate Python](http://book.pythontips.com/en/latest/)
3. [100 numpy exercises](https://github.com/rougier/numpy-100/blob/master/100%20Numpy%20exercises.md)
4. [awesome-python](https://github.com/vinta/awesome-python)

<br>
