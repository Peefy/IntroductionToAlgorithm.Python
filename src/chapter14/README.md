---
layout: post
title: "用Python实现红黑树数据结构的扩张"
description: "用Python实现红黑树数据结构的扩张"
categories: [Python]
tags: [python]
redirect_from:
  - /2018/04/07/
---

[红黑树代码](https://peefy.github.io/blog/2018/04/11/Python-RedBlackTree/)

## 用Python实现的红黑树RedBlackTree数据结构扩张的**顺序统计树OSTree**

OSTree 继承自 RedBlackTree

```python

class OSTree(RedBlackTree):
    '''
    顺序统计树
    '''
    def __init__(self):
        '''
        顺序统计树
        '''
        super().__init__()   

    def insert(self, z : OSTreeNode):
        '''
        插入顺序统计树结点
        '''
        super().insert(z)
        self.updatesize()

    def insertkey(self, key):
        '''
        插入顺序统计树结点
        '''
        node = OSTreeNode(key)
        self.insert(node)

    def leftrotate(self, x : RedBlackTreeNode):
        '''
        左旋 时间复杂度: `O(1)`
        '''
        y : RedBlackTreeNode = x.right
        z = y.left
        if y == self.nil:
            return 
        y.left.p = x
        y.p = x.p
        if x.p == self.nil:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.left = x
        x.p = y
        x.right = z
        y.size = x.size
        x.size = x.left.size + x.right.size + 1

    def rightrotate(self, x : RedBlackTreeNode):
        '''
        右旋 时间复杂度:`O(1)`
        '''
        y : RedBlackTreeNode = x.left
        z = y.right
        if y == self.nil:
            return
        y.right.p = x
        y.p = x.p
        if x.p == self.nil:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.right = x
        x.p = y
        x.left = z
        y.size = x.size
        x.size = x.left.size + x.right.size + 1

    def __os_select(self, x : RedBlackTreeNode, i):
        r = x.left.size + 1
        if i == r:
            return x
        elif i < r:
            return self.__os_select(x.left, i)
        else:
            return self.__os_select(x.right, i - r)

    def os_select(self, i):
        '''
        返回树中包含第`i`小关键字的结点的指针(递归)
        '''
        assert i >= 1 
        return self.__os_select(self.root, i)

    def os_select_nonrecursive(self, i):
        '''
        返回树中包含第`i`小关键字的结点的指针(递归)
        '''
        r = -1
        x = self.root
        while i != r:
            last = x 
            r = x.left.size + 1
            if i < r:
                x = x.left
            elif i > r:
                x = x.right
                i = i - r
        return last

    def os_rank(self, x : RedBlackTreeNode):
        '''
        对顺序统计树T进行中序遍历后得到的线性序中`x`的位置
        '''
        r = x.left.size + 1
        y = x
        while y != self.root:
            if y == y.p.right:
                r = r + y.p.left.size + 1
            y = y.p
        return r    

    def os_key_rank(self, key):
        '''
        对顺序统计树T进行中序遍历后得到的线性序中键值为`key`结点的位置
        '''
        node = self.tree_search(self.root, key)
        return self.os_rank(node)

    def __updatesize(self, x : RedBlackTreeNode):
        if x.isnil() == True:
            return 0
        x.size = self.__updatesize(x.left) + self.__updatesize(x.right) + 1
        return x.size

    def updatesize(self):
        '''
        更新红黑树的所有结点的size域
        '''
        self.__updatesize(self.root)

    @staticmethod
    def test():
        '''
        测试函数
        '''
        tree = OSTree()
        tree.insertkey(12)
        tree.insertkey(13)
        tree.insertkey(5)
        tree.insertkey(8)
        tree.insertkey(16)
        tree.insertkey(3)
        tree.insertkey(1)    
        tree.insertkey(2)
        print(tree.all())
        print(tree.os_select(1))
        print(tree.os_select(2))
        print(tree.os_select(3))
        print(tree.os_select(4))
        print(tree.os_select(5))
        print(tree.os_select(6))
        print(tree.os_key_rank(8))
        print(tree.os_key_rank(12))

```

## 用Python实现的红黑树RedBlackTree数据结构扩张的**区间树IntervalTree**

区间树IntervalTree 继承自 RedBlackTree

```python

__para_interval_err = 'para interval must be a tuple like ' + \
            'contains two elements, min <= max'

class IntervalTreeNode(RedBlackTreeNode):
    '''
    区间树结点
    '''
    def __init__(self, key, interval : tuple):
        '''
        区间树结点

        `key` : 键值

        `interval` : 区间值 a tuple like (`min`, `max`), and `min` <= `max`

        '''

        try:
            assert type(interval) is tuple
            assert len(interval) == 2 
            self.low, self.high = interval
            assert self.low <= self.high 
        except:
            raise Exception(__para_interval_err)           
        super().__init__(key)      
        self.interval = interval 
        self.max = 0

class IntervalTree(RedBlackTree):
    '''
    区间树
    '''
    def __init__(self):
        '''
        区间树
        '''
        super().__init__()
        
    def __updatemax(self, x : IntervalTreeNode):
        if x == None:
            return 0
        x.max = max(x.high, self.__updatemax(x.left), \
            self.__updatemax(x.right))
        return x.max
    
    def updatemax(self):
        '''
        更新区间树的`max`域
        '''
        self.__updatemax(self.root)

    def buildnil(self):
        '''
        构造一个新的哨兵nil结点
        '''
        nil = IntervalTreeNode(None, (0, 0))
        nil.color = BLACK
        nil.size = 0
        return nil

    def __int_overlap(self, int : tuple, i):
        low, high = int
        if i >= low and i <= high:
            return True
        return False

    def insert(self, x : IntervalTreeNode):
        '''
        将包含区间域`int`的元素`x`插入到区间树T中
        '''
        super().insert(x)
        self.updatemax()

    def delete(self, x : IntervalTreeNode):
        '''
        从区间树中删除元素`x`
        '''
        super().delete(x)
        self.updatemax()

    def interval_search(self, interval):
        '''
        返回一个指向区间树T中的元素`x`的指针，使int[x]与i重叠，
        若集合中无此元素存在，则返回`self.nil`
        '''
        x = self.root
        while x.isnil() == False and self.__int_overlap(x.interval, i) == False:
            if x.left.isnil() == False and x.left.max >= i.low:
                x = x.left
            else:
                x = x.right
        return x

    def insertkey(self, key, interval, index = None, color = RED):
        '''
        插入红黑树结点 时间复杂度 `O(lgn)`
        '''
        z = IntervalTreeNode(key, interval)
        self.insert(z)

    @staticmethod
    def test():
        '''
        test
        '''
        tree = IntervalTree()
        tree.insertkey(11, (0, 1))
        tree.insertkey(23, (1, 2))
        tree.insertkey(13, (2, 3))
        tree.insertkey(41, (0, 1))
        tree.insertkey(22, (1, 2))
        tree.insertkey(53, (2, 3))
        tree.insertkey(18, (0, 1))
        tree.insertkey(22, (1, 2))
        tree.insertkey(32, (2, 3))
        node = IntervalTreeNode(1, (1, 2))
        print(tree.all())

```

Test

```python

if __name__ == '__main__': 
    RedBlackTree.test()
    OSTree.test()
    IntervalTree.test()
    # python src/chapter14/rbtree.py
    # python3 src/chapter14/rbtree.py
else:
    pass

```

[Github Code](https://github.com/Peefy/CLRS_dugu_code-master/blob/master/src/chapter13)

