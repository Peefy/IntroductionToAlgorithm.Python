
from __future__ import absolute_import, print_function

from copy import deepcopy as _deepcopy

import time as _time
from random import randint as _randint

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

    def insertkey(self, key, index = None):
        '''
        插入元素，时间复杂度`O(h)` `h`为树的高度
        '''
        self.insert(SearchTreeNode(key, index))

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
            elif z.key > x.key:
                x = x.right
            else:
                # 处理相同结点的方式，随机分配左右结点
                if _randint(0, 1) == 0:
                    x = x.left
                else:
                    x = x.right
        z.p = y
        if y == None:
            self.root = z
        elif z.key < y.key:
            y.left = z
        elif z.key > y.key:
            y.right = z
        else:
            # 处理相同结点的方式，随机分配左右结点
            if _randint(0, 1) == 0:
                y.left = z
            else:
                y.right = z
        self.nodes.append(z) 
        self.lastnode = z

    def insertnodes(self, nodes : list):
        '''
        按顺序插入一堆结点
        '''
        for node in nodes:
            if node is type(SearchTreeNode):
                self.insert(node)
            else:
                self.insertkey(node)

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
        self.lastnode = z

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

    def leftrotate(self, x : SearchTreeNode):
        '''
        左旋 时间复杂度:`O(1)`
        '''
        if x.right == None:
            return
        y : SearchTreeNode = x.right
        x.right = y.left
        if y.left != None:
            y.left.p = x
        y.p = x.p
        if x.p == None:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.left = x
        x.p = y

    def rightrotate(self, x : SearchTreeNode):
        '''
        右旋 时间复杂度:`O(1)`
        '''
        if x.left == None:
            return
        y : SearchTreeNode = x.left
        x.left = y.right
        if y.right != None:
            y.right.p = x
        y.p = x.p
        if x.p == None:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.right = x
        x.p = y

class RandomSearchTree(SearchTree):

    def __init__(self):
        self.lastnode : SearchTreeNode = None
        self.root : SearchTreeNode = None
        self.nodes = []
        self.__buffers = []

    def __randomize_inplace(self, array):
        '''
        随机打乱排列一个数组

        Args
        ===
        `array` : 随机排列前的数组

        Return
        ===
        `random_array` : 随机排列后的数组

        '''
        n = len(array)
        for i in range(n):
            rand = _randint(i, n - 1)
            _time.sleep(0.001)
            array[i], array[rand] = array[rand], array[i]
        return array

    def randominsert(self, z : SearchTreeNode):
        '''
        使用随机化技术插入结点到缓存
        '''
        self.__buffers.append(z)

    def randominsertkey(self, key, index = None):
        '''
        使用随机化技术插入结点到缓存
        '''
        z = SearchTreeNode(key, index)
        self.randominsert(z)

    def update(self):
        '''
        从缓存更新二叉查找树结点
        '''
        randombuffers = self.__randomize_inplace(self.__buffers)
        for buffer in randombuffers:
            self.insert(buffer)
        self.__buffers.clear()
 
if __name__ == '__main__':
    tree = SearchTree()
    node1 = SearchTreeNode(12, 0)
    node2 = SearchTreeNode(11, 1)
    node3 = SearchTreeNode(10, 2)
    node4 = SearchTreeNode(15, 3)
    node5 = SearchTreeNode(9, 4)
    tree.insert_recursive(node1)
    tree.insert(node2)
    tree.insert(node3)
    tree.insert(node4)
    tree.insert_recursive(node5)   
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
    tree.insertkey(18)
    tree.insertkey(16)
    tree.leftrotate(node4)
    tree.insertkey(20)
    tree.rightrotate(node3)
    tree.insertkey(3)
    print(tree.all())
    random_tree = RandomSearchTree()
    random_tree.randominsertkey(1)
    random_tree.randominsertkey(2)
    random_tree.randominsertkey(3)
    random_tree.randominsertkey(4)
    random_tree.randominsertkey(5)
    random_tree.update()
    random_tree.insertkey(0)
    print(random_tree.all())
    print(random_tree.allkey())
    print(random_tree.inorder_tree_walk(random_tree.root))
    # python src/chapter12/searchtree.py
    # python3 src/chapter12/searchtree.py
else:
    pass