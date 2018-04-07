
from __future__ import absolute_import, print_function

from copy import deepcopy as _deepcopy

class SearchTreeNode:
    '''
    二叉查找树的结点
    '''
    def __init__(self, key, index, \
        leftindex = None, rightindex = None, \
        p = None, left = None, right = None):
        '''

        二叉树结点

        Args
        ===
        `left` : SearchTreeNode : 左儿子结点

        `right`  : SearchTreeNode : 右儿子结点

        `key` : 结点自身索引值

        `value` : 结点自身键值

        `leftkey` : 左儿子结点索引值

        `rightkey` : 右儿子结点索引值

        `p` : 父节点

        '''
        self.left = left
        self.right = right
        self.key = key
        self.index = index
        self.leftindex = leftindex
        self.rightindex = rightindex
        self.p = p

    def __str__(self):
        return 'key:' + str(self.key) + \
                'index:' + str(self.index)

class SearchTree:
    '''
    二叉查找树
    '''
    def __init__(self):
        self.lastnode : SearchTreeNode = None
        self.root : SearchTreeNode = None
        self.nodes = []

    def inorder_tree_walk(self, x : SearchTreeNode):
        array = []
        if x != None:
            left = self.inorder_tree_walk(x.left)
            array = array + left
            right = self.inorder_tree_walk(x.right)
            array = array + right
        if x != None:
            array.append(str(x))
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

    def minimum_recursive(self, x : SearchTreeNode):
        '''
        最小关键字元素(递归版本) 

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        if x != None and x.left != None:
            self.minimum_recursive(x.left)
        return x

    def maximum(self, x : SearchTreeNode):
        '''
        最大关键字元素(迭代版本)

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        while x.right != None:
            x = x.right
        return x
    
    def maximum_recursive(self, x : SearchTreeNode):
        '''
        最大关键字元素(递归版本)

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        while x != None and x.right != None:
            self.maximum_recursive(x.right)
        return x

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

    def insert_recursive(self, z : SearchTreeNode):
        '''
        插入元素(递归版本)，时间复杂度`O(h)` `h`为树的高度
        '''
        pass

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
        
    def update(self):
        pass

    def all(self) -> list:
        '''
        返回二叉查找树中所有结点索引值，键值构成的集合
        '''
        array = []
        for node in self.nodes:
            array.append({ "index":node.index,"key" : node.key})
        return array

    def count(self):
        return len(self.nodes)
