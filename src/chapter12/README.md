---
layout: post
title: "用Python实现的二叉查找树SearchTree"
description: "用Python实现的二叉查找树SearchTree"
categories: [Python]
tags: [python]
redirect_from:
  - /2018/04/07/
---

## 用Python实现的二叉查找树SearchTree

Module

```python

from __future__ import absolute_import, print_function

from copy import deepcopy as _deepcopy

```

首先定义二叉查找树节点

```python
class SearchTreeNode:
    '''
    二叉查找树的结点
    '''
    def __init__(self, key, index, \
        p = None, left = None, right = None):
        '''

        二叉树结点

        Args
        ===
        `left` : SearchTreeNode : 左儿子结点

        `right`  : SearchTreeNode : 右儿子结点

        `index` : 结点自身索引值

        `key` : 结点自身键值

        `p` : 父节点

        '''
        self.left = left
        self.right = right
        self.key = key
        self.index = index
        self.p = p

    def __str__(self):
        return 'key:' + str(self.key) + ','\
                'index:' + str(self.index)
```

定义二叉查找树

```python
class SearchTree:
    '''
    二叉查找树
    '''
    def __init__(self):
        self.lastnode : SearchTreeNode = None
        self.root : SearchTreeNode = None
        self.nodes = []

    def inorder_tree_walk(self, x : SearchTreeNode):
        '''
        从二叉查找树的`x`结点后序遍历
        '''
        array = []
        if x != None:
            left = self.inorder_tree_walk(x.left)
            array = array + left
            right = self.inorder_tree_walk(x.right)  
        if x != None:
            array.append(str(x))
            array = array + right
        return array

    def __inorder_tree_walk_key(self, x : SearchTreeNode):
        '''
        从二叉查找树的`x`结点后序遍历
        '''
        array = []
        if x != None:
            left = self.__inorder_tree_walk_key(x.left)
            array = array + left
            right = self.__inorder_tree_walk_key(x.right)  
        if x != None:
            array.append(x.key)
            array = array + right
        return array

    def tree_search(self, x : SearchTreeNode, key):
        '''
        查找 

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        try:
            if x != None and key == x.key:
                return x
            if key < x.key:
                return self.tree_search(x.left, key)
            else:
                return self.tree_search(x.right, key)            
        except :
            return None

    def iterative_tree_search(self, x : SearchTreeNode, key):
        '''
        查找的非递归版本

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        while x != None:
            if key < x.key:
                x = x.left
            elif key > x.key:
                x = x.right
            else:
                return x
        return x

    def minimum(self, x : SearchTreeNode):
        '''
        最小关键字元素(迭代版本) 

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        while x.left != None:
            x = x.left
        return x

    def __minimum_recursive(self, x : SearchTreeNode) -> list:
        '''
        最小关键字元素(递归版本) 

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = []
        if x != None:
            ex = self.__minimum_recursive(x.left)
            if ex == []:
                z = x
                array.append(z)
                return array
            else:
                array = array + ex
        return array

    def minimum_recursive(self, x : SearchTreeNode):
        '''
        最小关键字元素(递归版本) 

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = self.__minimum_recursive(x)
        if len(array) != 0:
            return array.pop()
        return None

    def maximum(self, x : SearchTreeNode):
        '''
        最大关键字元素(迭代版本)

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        while x.right != None:
            x = x.right
        return x
    
    def __maximum_recursive(self, x : SearchTreeNode):
        '''
        最大关键字元素(递归版本)

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = []
        if x != None:
            ex = self.__maximum_recursive(x.right)
            if ex == []:
                z = x
                array.append(z)
                return array
            else:
                array = array + ex
        return array

    def maximum_recursive(self, x : SearchTreeNode):
        '''
        最大关键字元素(递归版本)

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = self.__maximum_recursive(x)
        if len(array) != 0:
            return array.pop()
        return None

    def successor(self, x : SearchTreeNode):
        '''
        前趋:结点x的前趋即具有小于x.key的关键字中最大的那个

        时间复杂度：`O(h)`, `h`为树的高度
        
        '''
        if x.right != None:
            return self.minimum(x.right)
        y = x.p
        while y != None and x == y.right:
            x = y
            y = y.p
        return y

    def predecessor(self, x : SearchTreeNode):
        '''
        后继:结点x的后继即具有大于x.key的关键字中最小的那个

        时间复杂度：`O(h)`, `h`为树的高度
        
        '''
        if x.left != None:
            return self.maximum(x.left)
        y = x.p
        while y != None and x == y.left:
            x = y
            y = y.p
        return y
         
    def insert(self, z : SearchTreeNode):
        '''
        插入元素，时间复杂度`O(h)` `h`为树的高度
        '''
        y = None
        x = self.root
        while x != None:
            y = x
            if z.key < x.key:
                x = x.left
            else:
                x = x.right
        z.p = y
        if y == None:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z
        self.nodes.append(z) 

    def __insertfrom(self, z : SearchTreeNode, x : SearchTreeNode, lastparent : SearchTreeNode):
        if x != None:
            if z.key < x.key:
                self.__insertfrom(z, x.left, x)
            else:
                self.__insertfrom(z, x.right, x)
        else:
            z.p = lastparent
            if z.key < lastparent.key:
                lastparent.left = z
            else:
                lastparent.right = z

    def insert_recursive(self, z : SearchTreeNode):
        '''
        插入元素(递归版本)，时间复杂度`O(h)` `h`为树的高度
        '''
        if self.root == None:
            self.root = z
        else:  
            self.__insertfrom(z, self.root, None)
        self.nodes.append(z) 

    def delete(self, z : SearchTreeNode):
        '''
        删除操作，时间复杂度`O(h)` `h`为树的高度
        '''
        if z.left == None or z.right == None:
            y = z
        else:
            y = self.successor(z)
        if y.left != None:
            x = y.left
        else:
            x = y.right
        if x != None:
            x.p = y.p
        if y.p == None:
            self.root = x
        else:
            if y == y.p.left:
                y.p.left = x
            else:
                y.p.right = x
        if y != None:
            z.key = y.key
            z.index = _deepcopy(y.index)
        self.nodes.remove(z) 
        return y
        
    def all(self) -> list:
        '''
        返回二叉查找树中所有结点索引值，键值构成的集合
        '''
        array = []
        for node in self.nodes:
            array.append({ "index":node.index,"key" : node.key})
        return array

    def allkey(self) -> list:
        '''
        按升序的方式输出所有结点`key`值构成的集合
        '''
        return self.__inorder_tree_walk_key(self.root)

    def count(self) -> int:
        '''
        二叉查找树中的结点总数
        '''
        return len(self.nodes)
```

二叉查找树测试程序

```python
if __name__ == '__main__':
    tree = SearchTree()
    tree.insert_recursive(SearchTreeNode(12, 0))
    tree.insert(SearchTreeNode(11, 1))
    tree.insert(SearchTreeNode(10, 2))
    tree.insert(SearchTreeNode(15, 3))
    tree.insert_recursive(SearchTreeNode(9, 4))   
    print(tree.all())
    print(tree.count())
    print(tree.inorder_tree_walk(tree.root))
    print(tree.tree_search(tree.root, 15))
    print(tree.tree_search(tree.root, 8))
    print(tree.iterative_tree_search(tree.root, 10))
    print(tree.iterative_tree_search(tree.root, 7))
    print(tree.maximum(tree.root))
    print(tree.maximum_recursive(tree.root))
    print(tree.minimum(tree.root))
    print(tree.minimum_recursive(tree.root))
    print(tree.successor(tree.root))
    print(tree.predecessor(tree.root))
    # python src/chapter12/searchtree.py
    # python3 src/chapter12/searchtree.py
else:
    pass
```

[Github Code](https://github.com/Peefy/CLRS_dugu_code-master/blob/master/src/chapter12)

