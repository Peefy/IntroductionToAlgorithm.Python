---
layout: post
title: "用Python实现的红黑树RedBlackTree"
description: "用Python实现的红黑树RedBlackTree"
categories: [Python]
tags: [python]
redirect_from:
  - /2018/04/07/
---

## 用Python实现的红黑树RedBlackTree

redblacktree.py

```python


BLACK = 0
RED = 1

class RedBlackTreeNode:
    '''
    红黑树结点
    '''
    def __init__(self, key, index = None, color = RED, \
        p = None, left = None, right = None):
        '''
        红黑树树结点

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
        self.color = color
        self.p = p

    def __str__(self):
        '''
        str({'key' : self.key, 
            'index' : self.index, 
            'color' : self.color})
        '''
        return  str({'key' : self.key, 
            'index' : self.index, 
            'color' : self.color})

    def isnil(self):
        '''
        判断红黑树结点是否是哨兵结点
        '''
        if self.key == None and self.color == BLACK:
            return True
        return False

class RedBlackTree:
    '''
    红黑树
    '''
    def __init__(self):
        '''
        红黑树
        '''
        self.nodes = []
        self.nil = RedBlackTreeNode(None, color=BLACK)
        self.root = self.nil

    def insertkey(self, key, index = None, color = RED):
        '''
        插入红黑树结点 时间复杂度 `O(lgn)`
        '''
        z = RedBlackTreeNode(key, index, color)
        self.insert(z)

    def insert(self, z : RedBlackTreeNode):
        '''
        插入红黑树结点 时间复杂度 `O(lgn)`
        '''
        y = self.nil
        x = self.root
        while x != self.nil:
            y = x
            if z.key < x.key:
                x = x.left
            else:
                x = x.right
        z.p = y
        if y == self.nil:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z
        z.left = self.nil
        z.right = self.nil
        z.color = RED
        self.insert_fixup(z)
        self.nodes.append(z)

    def insert_fixup(self, z : RedBlackTreeNode):
        '''
        修正红黑树性质，结点重新旋转和着色
        '''
        while z.p.color == RED:
            if z.p == z.p.p.left:
                y = z.p.p.right
                if y.color == RED:
                    z.p.color = BLACK
                    y.color = BLACK
                    z.p.p.color = RED
                    z = z.p.p
                elif y.color == BLACK and z == z.p.right:
                    z = z.p
                    self.leftrotate(z)
                elif y.color == BLACK and z == z.p.left:
                    z.p.color = BLACK
                    z.p.p.color = RED
                    self.rightrotate(z.p.p)
            else:
                y = z.p.p.left
                if y.color == RED:
                    z.p.color = BLACK
                    y.color = BLACK
                    z.p.p.color = RED
                    z = z.p.p
                elif y.color == BLACK and z == z.p.right:
                    z = z.p
                    self.leftrotate(z)
                elif y.color == BLACK and z == z.p.left:
                    z.p.color = BLACK
                    z.p.p.color = RED
                    self.rightrotate(z.p.p)               
        self.root.color = BLACK    
        
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
            
    def all(self):
        '''
        返回红黑树中所有的结点
        '''
        return self.nodes

    def inorder_tree_walk(self, x : RedBlackTreeNode):
        '''
        从红黑树的`x`结点后序遍历
        '''
        array = []
        if x != None:
            left = self.inorder_tree_walk(x.left)
            array = array + left
            right = self.inorder_tree_walk(x.right)  
        if x != None and x.isnil() == False:
            array.append(str(x))
            array = array + right
        return array

if __name__ == '__main__':
    tree = RedBlackTree()
    tree.insertkey(41)
    tree.insertkey(38)
    tree.insertkey(31)
    tree.insertkey(12)
    tree.insertkey(19)
    tree.insertkey(8)
    tree.insertkey(1)
    print(tree.inorder_tree_walk(tree.root))
else:
    pass

```

[Github Code](https://github.com/Peefy/CLRS_dugu_code-master/blob/master/src/chapter13)

