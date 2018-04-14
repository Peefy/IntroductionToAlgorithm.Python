
from __future__ import absolute_import, print_function

from copy import deepcopy as _deepcopy

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

    def successor(self, x : RedBlackTreeNode):
        '''
        前趋:结点x的前趋即具有小于x.key的关键字中最大的那个

        时间复杂度：`O(h)`, `h=lgn`为树的高度
        
        '''
        if x.right != None:
            return self.minimum(x.right)
        y = x.p
        while y != None and x == y.right:
            x = y
            y = y.p
        return y

    def predecessor(self, x : RedBlackTreeNode):
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

    def minimum(self, x : SearchTreeNode):
        '''
        最小关键字元素(迭代版本) 

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        while x.left != None:
            x = x.left
        return x

    def __minimum_recursive(self, x : SearchTreeNode):
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
        插入元素后 修正红黑树性质，结点重新旋转和着色
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
        
    def delete_fixup(self, x : RedBlackTreeNode):
        '''
        删除元素后 修正红黑树性质，结点重新旋转和着色
        '''
        while x != self.root and x.color == BLACK:
            if x == x.p.left:
                w : RedBlackTreeNode = x.p.right
                if w.color == RED:
                    w.color = BLACK
                    x.p.color = RED
                    self.leftrotate(x.p)
                    w = x.p.right
                    if w.left.color == BLACK and w.right.color == BLACK:
                        w.color = RED
                        x = x.p
                    elif w.right.color == BLACK:
                        w.left.color = BLACK
                        w.color = RED
                        self.rightrotate(w)
                        w = x.p.right
                    else:
                        w.color = x.p.color
                        x.p.color = BLACK
                        w.right.color = BLACK
                        self.leftrotate(x.p)
                        x = self.root
            else:
                w : RedBlackTreeNode = x.p.left
                if w.color == RED:
                    w.color = BLACK
                    x.p.color = RED
                    self.rightrotate(x.p)
                    w = x.p.left
                    if w.right.color == BLACK and w.left.color == BLACK:
                        w.color = RED
                        x = x.p
                    elif w.left.color == BLACK:
                        w.right.color = BLACK
                        w.color = RED
                        self.leftrotate(w)
                        w = x.p.left
                    else:
                        w.color = x.p.color
                        x.p.color = BLACK
                        w.left.color = BLACK
                        self.rightrotate(x.p)
                        x = self.root
        x.color = BLACK

    def delete(self, z : RedBlackTreeNode):
        '''
        删除
        '''
        if z.left == self.nil or z.right == self.nil:
            y = z
        else:
            y = self.successor(z)
        if y.left != self.nil:
            x = y.left
        else:
            x = y.right
        x.p = y.p
        if x.p = self.nil:
            self.root = x
        elif y == y.p.left:
            y.p.left = x
        else:
            y.p.right = x
        if y != z:
            z.key = y.key
            z.index = _deepcopy(y.index)
        if y.color == BLACK:
            self.delete_fixup(x)
        return y
    
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