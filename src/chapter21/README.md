---
layout: post
title: "用Python实现的用于不相交集合的数据结构"
description: "用Python实现的用于不相交集合的数据结构"
categories: [Python]
tags: [python]
redirect_from:
  - /2018/06/18/
---

## 用于不相交集合的数据结构

不相交集合数据结构保持一组不相交的动态集合S={S1,S2,...,Sk}，每个集合通过一个代表来识别，代表即集合中的某个成员。集合中的每一个元素是由一个对象表示的，设x表示一个对象，希望支持以下操作

* MAKE-SET(x)：其唯一成员(因而其代表)就是x。因为各集合是不相交的，故要求x没有在其他集合出现过

* UNION(x, y)：将包含x和y的动态集合(比如说Sx和Sy)合并为一个新的集合(并集)

* FIND-SET(x)：返回一个指针，指向包含x的(唯一)集合的代表

不相交集合数据结构有多种应用,其中之一是用于确定一个无向图中连通子图的个数

使用*按秩结合*和*路径压缩*的启发式方法实现不相交集合森林，从渐进意义上来说是最快的方法

```python

class UndirectedGraph:
    '''
    无向图 `G=(V, E)`
    '''
    def __init__(self, vertexs : list = [], edges : list = []):
        '''
        无向图 `G=(V, E)`

        Args
        ===
        `vertexs` : 顶点集合 `list` contains element which contains one element denote a point

        `edges` : 边集合 `list` contains element which contains two elements denote one edge of two points repectively

        Example
        ===
        ```python
        import notintersectset as graph
        >>> g = graph.UndirectedGraph(['a', 'b', 'c', 'd'], [('a', 'b')])
        ```
        '''
        self.vertexs = vertexs
        self.edges = edges
        self.__findcount = 0
        self.__unioncount = 0
        self.__kcount = 0

    def get_connected_components(self):
        '''
        获取无向图中连通子图的集合
        '''
        self.__findcount = 0
        self.__unioncount = 0
        self.__kcount = 0
        set = Set()
        for v in self.vertexs:
            set.make_set(v)
        for e in self.edges:
            u, v = e
            set1 = set.find(u)
            set2 = set.find(v)
            self.__findcount += 2
            if set1 != set2:
                set.union(set1, set2)
                self.__unioncount += 1
        self.__kcount = len(set.sets)
        return set

    def print_last_connected_count(self):
        '''
        获取上一次连接无向图之后调用函数情况
        '''
        print('the k num:{} the find num:{} the union num:{}'. \
            format(self.__kcount, self.__findcount, self.__unioncount))

class Set:
    '''
    不相交集合数据结构
    '''
    def __init__(self):
        '''
        不相交集合数据结构
        '''
        self.sets = []

    def make_set(self, element):
        '''
        用元素`element`建立一个新的集合
        '''
        self.sets.append({element})

    def union(self, set1, set2):
        '''
        将子集合`set1`和`set2`合并
        '''
        if set1 is None or set2 is None:
            return
        self.sets.remove(set1)
        self.sets.remove(set2)
        self.sets.append(set1 | set2)

    def find(self, element):
        '''
        找出包含元素`element`的集合
        '''
        for set in self.sets:
            if element in set:
                return set
        return None
    
    def __str__(self):
        return str(self.sets)

    def printsets(self):
        '''
        打印集合
        '''
        for set in self.sets:
            print(set)

class ListNode:
    def __init__(self, key = None):
        '''
        采用链表表示不相交集合结点
        '''
        self.first = None
        self.next = None
        self.key = key
    
    def __str__(self):
        return str(self.key)

class List:
    def __init__(self):
        '''
        采用链表表示不相交集合
        '''
        self.rep = None
        self.head = None
        self.tail = None
        self.size = 0
    
    def __str__(self):
        return 'List size:{} and rep:{}'.format(self.size, self.rep)

class ListSet(Set):
    '''
    不相交集合的链表表示
    '''
    def __init__(self):
        '''
        不相交集合的链表表示
        '''
        self.sets = []

    def make_set(self, element):
        '''
        用元素`element`建立一个新的集合
        '''
        list = List()
        node = ListNode(element)  
        if list.size == 0:           
            list.head = node
            list.tail = node
            list.rep = node
            node.first = node
            list.size = 1
        else:
            list.tail.next = node
            list.tail = node 
            node.first = list.head
            list.size += 1
        self.sets.append(list)

    def union(self, set1, set2):
        '''
        将子集合`set1`和`set2`合并
        '''
        self.sets.remove(set1)
        self.sets.remove(set2)
        set1.tail.next = set2.rep
        set1.size += set2.size
        set1.tail = set2.tail

        set2.rep = set1.rep

        node = set2.head
        for i in range(set2.size):
            node.first = set1.rep
            node = node.next

        self.sets.append(set1)

    def unionelement(self, element1, element2):
        '''
        将`element1`代表的集合和`element2`代表的集合合并
        '''
        set1 = self.find(element1)
        set2 = self.find(element2)
        if set1 is None or set2 is None:
            return
        if set1.size < set2.size:
            self.union(set2, set1)
        else:
            self.union(set1, set2)

    def find(self, element):
        '''
        找出包含元素`element`的集合
        '''
        for set in self.sets:
            node = set.rep
            while node != set.tail:
                if node.key == element:
                    return set
                node = node.next
            else:
                if set.tail.key == element: 
                    return set
        return None

class RootTreeNode:
    '''
    有根树结点
    '''
    def __init__(self, key = None, parent = None, rank = None):
        '''
        有根树结点

        Args
        ===
        `key` : 关键字值

        `parent` : 结点的父结点

        `rank` : 结点的秩
        '''
        self.key = key
        self.parent = parent
        self.rank = rank
    
    def __str__(self):
        return 'key:{} rank:{}'.format(self.key, self.rank)

class RootTree:
    '''
    有根树
    '''
    def __init__(self, root = None):
        '''
        有根树
        Args
        ===
        `root` : 有根树的根结点
        '''
        self.root = root
    
    def __str__(self):
        return 'roots:' + str(self.root)

class ForestSet(Set):
    '''
    不相交集合森林
    '''
    def __init__(self):
        self.sets = []

    def make_set(self, element):
        '''
        用元素`element`建立一个新的集合
        '''
        treenode = RootTreeNode(element)
        self.make_set_node(treenode)
        
    def make_set_node(self, node : RootTreeNode):
        '''
        用有根树结点`node`建立一个新的集合
        '''
        tree = RootTree()  
        node.parent = node
        node.rank = 0
        tree.root = node
        self.sets.append(tree)

    @classmethod
    def link(self, node1 : RootTreeNode, node2 : RootTreeNode):
        '''
        连接两个有根树的结点`node1`和`node2`
        '''
        if node1.rank > node2.rank:
            node2.parent = node1
        else:
            node1.parent = node2
            if node1.rank == node2.rank:
                node2.rank += 1
    
    def union(self, x : RootTreeNode, y : RootTreeNode):
        '''
        将有根树结点`x`代表的集合和有根树结点`y`代表的集合合并
        '''
        self.link(self.findnode(x), self.findnode(y))

    def findnode(self, x : RootTreeNode):
        '''
        带路径压缩的寻找集合
        '''
        if x != x.parent:
            x.parent = self.findnode(x.parent)
        return x.parent

    def findnode_nonrecursive(self, x : RootTreeNode):
        '''
        带路径压缩的寻找集合(非递归版本)
        '''
        y = x
        while y != y.parent:
            y = y.parent
        while x != x.parent:
            x.parent = y
            x = x.parent

def connected_components(g: UndirectedGraph):
    '''
    求一个无向图中连通子图的个数

    Args
    ===
    `g` : UndirectedGraph 无向图

    '''
    set = Set()
    for v in g.vertexs:
        set.make_set(v)
    for e in g.edges:
        u, v = e
        set1 = set.find(u)
        set2 = set.find(v)
        if set1 != set2:
            set.union(set1, set2)
    return set

def test_graph_connected():
    '''
    测试无向图链接连通子图
    '''
    g = UndirectedGraph()
    g.vertexs = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    g.edges.append(('b', 'd'))
    g.edges.append(('e', 'g'))
    g.edges.append(('a', 'c'))
    g.edges.append(('h', 'i'))
    g.edges.append(('a', 'b'))
    g.edges.append(('e', 'f'))
    g.edges.append(('b', 'c'))
    print(g.get_connected_components())
    g.print_last_connected_count()

def test_list_set():
    '''
    不相交集合的链表表示
    '''
    NUM = 16
    set = ListSet()
    for i in range(NUM):
        set.make_set(i)
    for i in range(0, NUM - 1, 2):
        set.unionelement(i, i + 1)
    for i in range(0, NUM - 3, 4):
        set.unionelement(i, i + 2)
    set.printsets()
    set.unionelement(1, 5)
    set.unionelement(11, 13)
    set.unionelement(1, 10)
    set.printsets()
    print(set.find(2))
    print(set.find(9))

def test_forest_set():
    '''
    测试不相交集合森林
    '''
    NUM = 16
    set = ForestSet()
    nodes = []
    for i in range(NUM):
        nodes.append(RootTreeNode(i))
    for i in range(NUM):
        set.make_set_node(nodes[i])
    set.printsets()
    for i in range(0, NUM - 1, 2):
        set.union(nodes[i], nodes[i + 1])
    set.printsets()
    for i in range(0, NUM - 3, 4):
        set.union(nodes[i], nodes[i + 2])
    set.printsets()

if __name__ == '__main__':
    test_graph_connected()
    test_list_set()
    test_forest_set()
else:
    pass    

```

[Github Code](https://github.com/Peefy/IntroductionToAlgorithm.Python/blob/master/src/chapter21)
