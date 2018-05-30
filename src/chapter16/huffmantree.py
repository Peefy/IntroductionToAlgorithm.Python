
from __future__ import division, absolute_import, print_function
from copy import deepcopy as _deepcopy

class HuffmanTreeNode:
    '''
    Huffman二叉树结点
    '''
    def __init__(self, left = None, right = None, f = None, p = None, character=None, index=None):
        '''
        Huffman二叉树结点

        Args
        ===
        `left` : BTreeNode : 左儿子结点

        `right`  : BTreeNode : 右儿子结点

        `f` : 结点自身频度

        '''
        self.left = left
        self.right = right
        self.f = f
        self.p = p
        self.character = character
        self.coding = ''
        self.index = None

class HuffmanTree:
    def __init__(self):
        self.root = None
        self.__nodes = []
        self.codings = []
        self.characters = []
        self.fs = []
        self.__coding = ""
        
    def addnode(self, node):
        '''
        加入二叉树结点

        Args
        ===
        `node` : `HuffmanTreeNode` 结点

        '''
        self.__nodes.append(node)

    def buildnodecodingformcharacter(self, node):
        if node is not None:
            if node.p is None:
                return
            if node.p.left == node:
                self.__coding += '0'
            if node.p.right == node:
                self.__coding += '1'
            self.buildnodecodingformcharacter(node.p)
        
    def __findnode(self, f):
        '''
        根据`f`从`nodes`中寻找结点
        '''
        if f is None:
            return None
        for node in self.__nodes:
            if f == node.f:
                return node
            if node.left is not None:
                if f == node.left.f:
                    return node.left
            if node.right is not None:
                if f == node.right.f:
                    return node.right
        return None

    def __findnode_f_c(self, f, c):
        '''
        根据`f`从`nodes`中寻找结点
        '''
        if f is None:
            return None
        for node in self.__nodes:
            if f == node.f and c == node.character:
                return node
            if node.left is not None:
                if f == node.left.f and c == node.left.character:
                    return node.left
            if node.right is not None:
                if f == node.right.f and c == node.right.character:
                    return node.right
        return None

    def __findnodefromc(self, c):
        '''
        根据`f`从`nodes`中寻找结点
        '''
        if c is None:
            return None
        for node in self.__nodes:
            if c == node.character:
                return node
            if node.left is not None:
                if c == node.left.character:
                    return node.left
            if node.right is not None:
                if c == node.right.character:
                    return node.right
        return None

    def renewall(self):
        '''
        更新/连接/构造二叉树
        '''
        for node in self.__nodes:
            if node.left is not None:
                node.left = self.__findnode_f_c(node.left.f, node.left.character)
                node.left.p = node
            if node.right is not None:
                node.right = self.__findnode_f_c(node.right.f, node.right.character)
                node.right.p = node
    
    def renewnode(self, node):
        '''
        更新/连接/构造二叉树结点
        '''
        if node is None:
            return
        if node.left is not None:
            node.left = self.__findnode_index(node.left.index)
            node.left.p = node
        if node.right is not None:
            node.right = self.__findnode_index(node.right.index)
            node.right.p = node

    def renewallcoding(self, characters):
        n = len(characters)
        for i in range(n):
            c = characters[i]
            node = self.__findnodefromc(c)
            self.__coding = ""
            self.buildnodecodingformcharacter(node)
            if node is not None:
                node.coding = self.__coding[::-1]
                self.codings.append(node.coding)

class HuffmanTreeBuilder:
    '''
    HuffmanTree 构造器
    '''
    def __init__(self, C : list, f : list):
        self.C = C
        self.f = f

    def extract_min(self, C : list):
        min_val = min(C)
        C.remove(min_val)
        node = HuffmanTreeNode(None, None, min_val)
        return node

    def build_character(self, C : list, f : list, node : HuffmanTreeNode):
        try:
            index = f.index(int(node.f))
            f.pop(index)
            node.character = C[index]
            C.pop(index)
        except Exception as err:
            pass

    def huffman(self, C : list, f : list):
        '''
        赫夫曼编码

        算法自底向上的方式构造出最优编码所对应的树T

        Args
        ===
        `C` : 一个包含n个字符的集合，且每个字符都是一个出现频度为f[c]的对象

        '''
        n = len(f)
        Q = _deepcopy(f)
        tree = HuffmanTree()
        tree.characters = _deepcopy(self.C)
        tree.fs = _deepcopy(self.f)
        index = 0
        for i in range(n - 1):
            x = self.extract_min(Q)
            self.build_character(C, f, x)
            x.index = index
            index += 1
            y = self.extract_min(Q)
            self.build_character(C, f, y)
            y.index = index
            index += 1
            z = HuffmanTreeNode(x, y, x.f + y.f)
            z.index = index
            index += 1
            x.p = z
            y.p = z
            tree.addnode(z)
            Q.append(z.f)
        tree.renewall()
        tree.root = z
        tree.renewallcoding(tree.characters)
        
        return tree

    def build(self):
        '''
        构造一个HuffmanTree
        '''
        return self.huffman(self.C, self.f)
        
