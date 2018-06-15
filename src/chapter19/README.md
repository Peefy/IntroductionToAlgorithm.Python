---
layout: post
title: "用Python实现二项树"
description: "用Python实现的二项树"
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
    def __init__(self, key=None, degree=None, p=None, 
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
        '''
        str(self.key)
        '''
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
        min = 2147483648
        while x != None:
            if x.key < min:
                min = x.key
                y = x
            x = x.sibling
        return y

    @classmethod
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

    def insert(self, x : BinomialHeapNode):
        '''
        插入一个结点 时间复杂度`O(1) + O(lgn) = O(lgn)`
        '''
        h1 = BinomialHeap.make_heap()
        x.p = None
        x.child = None
        x.sibling = None
        x.degree = 0
        h1.head = x
        unionheap = BinomialHeap.union(self, h1)
        return unionheap

    def insertkey(self, key):
        '''
        插入一个结点 时间复杂度`O(1) + O(lgn) = O(lgn)`
        '''
        return self.insert(BinomialHeapNode(key))

    def extract_min(self):
        '''
        抽取最小关键字
        '''
        p = self.head
        x = None
        x_prev = None
        p_prev = None
        min = -2147483648
        if p is None:
            return p
        x = p
        min = p.key
        p_prev = p
        p = p.sibling
        while p is not None:
            if p.key < min:
                x_prev = p_prev
                x = p
                min = p.key
            p_prev = p
            p = p.sibling
        if x == self.head:
            self.head = x.sibling
        elif x.sibling is None:
            x_prev.sibling = None
        else:
            x_prev.sibling = x.sibling
        child_x = x.child
        if child_x != None:
            h1 = BinomialHeap.make_heap()
            child_x.p = None
            h1.head = child_x
            p = child_x.sibling
            child_x.sibling = None
            while p is not None:
                p_prev = p
                p = p.sibling
                p_prev.sibling = h1.head
                h1.head = p_prev
                p_prev.p = None
            self = BinomialHeap.union(self, h1)         
        return self

    def decresekey(self, x : BinomialHeapNode, key):
        '''
        减小结点的关键字的值，调整该结点在相应二项树中的位置
        '''
        if x.key < key:
            return 
        x.key = key
        y = x
        p = x.p
        while p is not None and y.key < p.key:
            y.key = p.key
            p.key = key
            y = p
            p = y.p

    def delete(self, x : BinomialHeapNode):
        '''
        删除一个关键字
        '''
        self.decresekey(x, -2147483648)
        return self.extract_min()

    @classmethod
    def make_heap(self):
        '''
        创建一个新的二项堆
        '''
        heap = BinomialHeap()
        return heap

    @classmethod
    def merge(self, h1, h2):
        '''
        合并两个二项堆h1和h2
        '''
        firstNode = None
        p = None
        p1 = h1.head
        p2 = h2.head
        if p1 is None or p2 is None:
            if p1 is None:
                firstNode = p2
            else:
                firstNode = p1
            return firstNode
        if p1.degree < p2.degree:
            firstNode = p1
            p = firstNode
            p1 = p1.sibling
        else:
            firstNode = p2
            p = firstNode
            p2 = p2.sibling
        while p1 is not None and p2 is not None:
            if p1.degree < p2.degree:
                p.sibling = p1
                p = p1
                p1 = p1.sibling
            else:
                p.sibling = p2
                p = p2
                p2 = p2.sibling
        if p1 is not None:
            p.sibling = p1
        else:
            p.sibling = p2
        return firstNode
    
    @classmethod
    def union(self, h1, h2):
        '''
        两个堆合并 时间复杂度:`O(lgn)`
        '''
        h = BinomialHeap.make_heap()
        h.head = BinomialHeap.merge(h1, h2)
        del h1
        del h2
        if h.head is None:
            return h
        prev = None
        x = h.head
        next = x.sibling
        while next is not None:
            if x.degree != next.degree or (next.sibling is not None and x.degree == next.sibling.degree):
                prev = x
                x = next
            elif x.key <= next.key:
                x.sibling = next.sibling
                h.link(next, x)
            else:
                if prev is None:
                    h.head = next
                else:
                    prev.sibling = next
                h.link(x, next)
            next = x.sibling
        return h
                
def test():
    '''
    test
    '''
    print('BinomialHeapNode and BinomialHeap test')
    heap = BinomialHeap.make_heap()

    node = BinomialHeapNode(5)
    heap = heap.insert(node)
    heap = heap.insertkey(8)
    heap = heap.insertkey(2)
    heap = heap.insertkey(7)
    heap = heap.insertkey(6)
    heap = heap.insertkey(9)
    heap = heap.insertkey(4)
    heap = heap.extract_min()
    print(heap.minimum())
    heap = heap.delete(node)
    if heap.head is not None:
        print(heap.head)
    heap = heap.extract_min()
    if heap.head is not None:
        print(heap.head)

if __name__ == '__main__':
    test()
else:
    pass

```

[Github Code](https://github.com/Peefy/CLRS_dugu_code-master/blob/master/src/chapter19)
