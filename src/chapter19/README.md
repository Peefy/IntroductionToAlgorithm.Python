---
layout: post
title: "用Python实现二项堆"
description: "用Python实现的二项堆"
categories: [Python]
tags: [python]
redirect_from:
  - /2018/06/15/
---

## 二项树

二项树Bk是一种递归定义的树。

二项树B0只含包含一个结点。二项树Bk由两颗二项树Bk-1链接而成：其中一棵树的根的是另一棵树的根的最左孩子

## 二项堆

二项堆H由一组满足下面的二项堆性质的二项树组成

(1) H中的每个二项树遵循最小堆性质：结点的关键字大于或等于其父结点的关键字,我们说这种树是最小堆有序的

(2) 对任意非负整数k，在H中至多有一棵二项树的根具有度数k

```python

class BinomialHeapNode:
    '''
    二项堆结点
    '''
    def __init__(self, p = None, key = None, degree = None, 
        child = None, sibling = None):
        '''
        二项堆结点

        Args
        ===
        `p` : 父结点
        
        `key` : 关键字

        `degree` : 子结点的个数

        `child` : 子结点

        `sibling` : 二项堆同根的下一个兄弟

        '''
        self.p = p
        self.key = key
        self.degree = degree
        self.child = child
        self.sibling = sibling
    
    def __str__(self):
        return str(self.key)

class BinomialHeap:
    '''
    二项堆
    '''
    def __init__(self, head : BinomialHeapNode = None):
        '''
        二项堆

        Args
        ===
        `head` : 头结点

        '''
        self.head = head

    def minimum(self):
        '''
        求出指向包含n个结点的二项堆H中具有最小关键字的结点

        时间复杂度`O(lgn)`

        '''
        y = None
        x = self.head
        min = -2147483648
        while x != None:
            if x.key < min:
                min = x.key
                y = x
            x = x.sibling
        return y

    def link(self, y : BinomialHeapNode, z : BinomialHeapNode):
        '''
        将一结点`y`为根的Bk-1树与以结点`z`为根的Bk-1树连接起来
        也就是使`z`成为`y`的父结点，并且成为一棵Bk树

        时间复杂度`O(1)`

        Args
        ===
        `y` : 一个结点

        `z` : 另外一个结点
        '''
        y.p = z
        y.sibling = z.child
        z.child = y
        z.degree += 1

    def extract_min(self):
        '''
        '''
        pass

    @classmethod
    def make_heap(self):
        '''
        创建一个新的二项堆
        '''
        heap = BinomialHeap()
        return heap

    @classmethod
    def merge(self, H1, H2):
        pass

    @classmethod
    def union(self, H1, H2):
        '''
        两个堆合并
        '''
        pass
        
```

[Github Code](https://github.com/Peefy/CLRS_dugu_code-master/blob/master/src/chapter19)
