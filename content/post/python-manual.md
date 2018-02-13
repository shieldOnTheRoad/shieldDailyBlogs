---
title: "Python Manual"
date: 2018-02-06T12:48:36+08:00
lastmod: 2018-02-11T15:48:36+08:00
draft: false
tags: ["python", "code"]
categories: ["Program"]
author: 'shield'

weight: 10

# You can also close(false) or open(true) something for this content.
# P.S. comment can only be closed
# comment: false
# autoCollapseToc: false
# reward: falsed
toc: false
mathjax: true
---

我的`python`速查手册。计算机语言这玩意，经久不用就容易忘记，记录下来。

<!--more-->
<br>

## #0 **CHEAT SHEET**
| api           | description        | category          |
|:--------------|:-------------------|:------------------|
| np.loadtxt('file.txt', delimiter=',') | load .txt file | <font face='Times New Roman' color='#6699ff'>IMPORTING / EXPORTING</font> |
| np.genfromtxt('file.csv', delimiter=',') | load .csv file | |
| np.savetxt('file.txt', arr, delimiter=',') | writes to a text file | |
| np.zeros(3) | 3x1 arr with all values 0 | <font face='Times New Roman' color='#6699ff'>CREATING ARRAYS</font> |
| np.ones((3,4)) | 3x4 arr with all values 1 | |
| np.eye(5) | 5x5 array of Identity matrix| |
| np.diag(np.arange(1,4), k=0) | create a diagonal matrix | |
| np.linspace(0, 100, 6) | 6 uniform values from 0 to 100 | |
| np.full((2,3), 8) | 2x3 arr with all values 8 | |
| np.random.rand(4, 5) | 4x5 arr of random val in [0,1) | |
| np.random.randint(5, size=(2,3)) | 2x3 arr of random ints in [0, 5) | |
| np.copy(arr) | copies arr to new memory | <font face='Times New Roman' color='#6699ff'>COPY / SORT / RESHAPE</font> |
| arr.sort(axis=0) | sorts specific axis of arr. | |
| two_d_arr.flatten() | flatten 2D array to 1D | |
| arr.reshape(3, 4) | reshape arr to 3x4 | |
| arr.resize((5,6)) | change arr to 5x6  | |
| np.tile(np.array([[0,1], [1,0]]), (4,4)) | create a 8x8 checkerboard | |
| np.append(arr, value) | append value to end of arr | <font face='Times New Roman' color='#6699ff'>ADD / RM ELEMENTS</font> |
| np.insert(arr, 2, value) | inserts val into arr. before idx2 | |
| np.delete(arr, 3, axis=0) | deletes row on idx3 of arr | |
| np.concatenate((arr1,arr2), axis=0) | add arr2 as rows to arr1 | <font face='Times New Roman' color='#6699ff'>COMBINING / SPLIT</font> |
| np.concatenate((arr1,arr2), axis=1) | add arr2 as columns to arr1 | |
| np.split(arr, 3) | split arr into 3 sub-arr | |
| np.hsplit(arr, 5) | split arr horizontally into 5 arr | |
| arr<5 | return an arr with boolean val | <font face='Times New Roman' color='#6699ff'>INDEX / SLICE / SUBSET</font> |
| (arr1<3) & (arr2>5) | return an arr with boolean val | |
| ~arr | invert a boolean arr | |
| arr[1::2] | select val from idx1 with step2 | |
| np.nonzero(arr) | find idxs of ~0 val from arr | |
| np.pad(arr, pad_width, mode) | add a border around arr | |
| np.unravel_index(10, (6,7,8)) |  the index of the 10th element | |
| np.ceil(arr) | round up to the nearest int | <font face='Times New Roman' color='#6699ff'>VECTOR MATH</font> |
| np.floor(arr) | round down to the nearest int | |
| np.round(arr) | round to the nearest int | |
| arr.size * arr.itemsize | find memory size of any arr | <font face='Times New Roman' color='#6699ff'>INFO STATISTICS</font> |
| np.info(arr) | get info of arr | |
| print(0.3 == 3 * 0.1) | return False :( | <font face='Times New Roman' color='#6699ff'>NOTE THAT</font> |


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

#### **2.2 virtualenv**
新建python虚拟环境。
```shell
$ pip install virtualenv
$ # virtualenv -p /usr/local/bin/python3.4 ENV3.4
$ # --system-site-packages继承原有所有包
$ virtualenv --no-site-package venv
$ source ./bin/activate  # 或者source ./Scripts/activate
$ pip list  # 查看当前环境依赖包信息
$ ./bin/deactivate  # 停止使用虚拟程序
```

<br>

## # **参考资料**
1. [Python进阶](http://docs.pythontab.com/interpy/)
2. [Intermediate Python](http://book.pythontips.com/en/latest/)
3. [100 numpy exercises](https://github.com/rougier/numpy-100/blob/master/100%20Numpy%20exercises.md)
4. [awesome-python](https://github.com/vinta/awesome-python)
5. [Data Science Cheat Sheet - NumPy](http://t.cn/RXKw3Ui)

<br>
