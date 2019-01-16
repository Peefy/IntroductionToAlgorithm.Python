---
layout: post
title: "用Python实现的图的基本算法"
description: "用Python实现的图的基本算法"
categories: [Python]
tags: [python]
redirect_from:
  - /2018/06/30/
---

## 图算法及应用

* 图的两种计算机表示法：邻接表和邻接矩阵
* 广度优先的图搜索算法 O(V+E)
* 深度优先的图搜索算法 O(V+E)
* DFS应用：有向无回路图的拓扑排序、O(V+E)
有向图的强连通分支SCC、
分支定界法：是案例不是方法
* 单源最短路径
* 任意两点间的最短路径
* 大数据下的图

## 邻接矩阵和邻接表

邻接表所需要的存储容量为O(V+E),邻接矩阵所需要的存储容量为O(V^2)

邻接表的不足：确定边(u,v)是否存在,智能在顶点u的邻接表Adj[u]中搜索v而没有其他更快的办法

## 广度优先搜索

已知图G=(V,E)和一个源顶点s,广度优先搜索系统地探寻G的边，从而发现s所能到达的所有顶点,并计算s到所有这些顶点的距离(最少边数)

## 图的基本算法

基于广度优先或深度优先图搜索的算法

## 图的表示

要表示一个有向图或者无向图G=(V,E),有两种标准的方法，即邻接表和邻接矩阵
这两种表示法即可以用于有向图，也可以用于无向图
通常采用邻接表表示法，因为用这种方法表示稀疏图比较紧凑

## 广度优先搜索

广度优先搜素是最简单的图搜索算法之一,也是很多重要的图算法的原型

## 深度优先搜索

深度搜索算法遵循的搜索策略是尽可能"深"地搜索一个图

在深度优先搜索中，对于最新发现的顶点，如果还有以此为起点而未探测到的边，就沿此边继续探测下去

## 拓扑排序

对有向图或者无向图G=(V,E)进行拓扑排序后，结果为该图所有顶点的一个线性序列

## 强连通分支

寻找图G=(V,E)的强连通分支的算法中使用了G的转置，其定义为G^T,建立G^T所需的时间为O(V+E)，G和G^T有着完全相同的强连通分支，即在G中u和v互为可达当且仅当在G^T中它们互为可达

1.调用DFS(G)计算出每个结点u的完成时刻f[u]
2.计算出G^T,
3.调用DFS(G^T),但在DFS的主循环里按f[u]递减的顺序考虑各结点
4.输出第2步中产生的深度优先森林中每棵树的顶点，作为各自独立的强连通分支

每棵深度优先树都是一个强连通分支

## 分支定界法

分支定界法是一种用途非常广的算法，也是一种技巧性很强的算法，不同的问题解法各不相同。

同顺序加工任务安排问题

## min-max搜索法


```python


import math as _math
from copy import deepcopy as _deepcopy

import numpy as _np

COLOR_WHITE = 0
COLOR_GRAY = 1
COLOR_BLACK = 2

DIRECTION_NONE = ' '
DIRECTION_TO = '→'
DIRECTION_FROM = '←'
DIRECTION_BOTH = '←→'

class Vertex:
    '''
    图的顶点
    '''
    def __init__(self, key = None):
        '''
        图的顶点

        Args
        ===
        `key` : 顶点关键字

        '''
        self.key = key
        self.color = COLOR_WHITE
        self.d = _math.inf
        self.pi = None
        self.f = _math.inf
        self.weightkey = 0

    def resetpara(self):
        '''
        复位所有属性
        '''
        self.color = COLOR_WHITE
        self.d = _math.inf
        self.pi = None
        self.f = _math.inf

    def __hash__(self):
        code = self.color.__hash__()
        code = code * 37 + self.key.__hash__()
        code = code * 37 + self.pi.__hash__()
        code = code * 37 + self.d.__hash__()
        code = code * 37 + self.f.__hash__()
        return code 

    def __str__(self):
        return '[key:{} color:{} d:{} f:{} pi:{}]'.format(self.key, \
            self.color, self.d, self.f, self.pi)

    def __lt__(self, other):
        if type(other) is Vertex:
            return self.weightkey < other.weightkey
        else:
            return self.weightkey < other

    def __gt__(self, other):
        if type(other) is Vertex:
            return self.weightkey > other.weightkey
        else:
            return self.weightkey > other

    def __le__(self, other):
        if type(other) is Vertex:
            return self.weightkey <= other.weightkey
        else:
            return self.weightkey <= other

    def __ge__(self, other):
        if type(other) is Vertex:
            return self.weightkey >= other.weightkey
        else:
            return self.weightkey >= other

    def __eq__(self, other):
        if type(other) is Vertex:
            return self.weightkey == other.weightkey
        else:
            return self.weightkey == other

    def __ne__(self, other):
        if type(other) is Vertex:
            return self.weightkey != other.weightkey
        else:
            return self.weightkey != other

class Edge:
    '''
    图的边，包含两个顶点
    '''
    def __init__(self, vertex1 : Vertex = None, \
            vertex2 : Vertex = None, \
            weight = 1, \
            dir = DIRECTION_NONE,
            ):
        '''
        图的边，包含两个顶点

        Args
        ===
        `vertex1` : 边的一个顶点
        
        `vertex2` : 边的另一个顶点

        `dir` : 边的方向   
            DIRECTION_NONE : 没有方向
            DIRECTION_TO : `vertex1` → `vertex2`
            DIRECTION_FROM : `vertex1` ← `vertex2`
            DIRECTION_BOTH : `vertex1` ←→ `vertex2`

        '''
        self.dir = dir
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.weight = weight

    def __str__(self):
        return str((self.vertex1.key, self.vertex2.key, self.dir, self.weight))

    def __hash__(self):
        code = self.dir.__hash__()
        code = code * 37 + self.vertex1.__hash__()
        code = code * 37 + self.vertex2.__hash__()
        code = code * 37 + self.weight.__hash__()
        return code

    def __lt__(self, other):
        if type(other) is Graph:
            return self.weight < other.weight 
        else:
            return self.weight < other

    def __gt__(self, other):
        if type(other) is Graph:
            return self.weight > other.weight 
        else:
            return self.weight > other

    def __le__(self, other):
        if type(other) is Graph:
            return self.weight <= other.weight
        else:
            return self.weight <= other

    def __ge__(self, other):
        if type(other) is Graph:
            return self.weight >= other.weight
        else:
            return self.weight >= other

    def __eq__(self, other):
        if type(other) is Graph:
            return self.weight == other.weight
        else:
            return self.weight == other

    def __ne__(self, other):
        if type(other) is Graph:
            return self.weight != other.weight
        else:
            return self.weight != other

class Graph:
    '''
    图`G=(V,E)`
    '''
    def __init__(self, vertexs : list = [], edges : list = []):
        '''
        图`G=(V,E)`

        Args
        ===
        `vertexs` : 图的顶点

        `edges` : 图的边

        '''
        self.veterxs = vertexs
        self.edges = edges
   
    def clear(self):
        '''
        清除所有顶点和边
        '''
        self.veterxs = []
        self.edges = []

    def hasdirection(self):
        '''
        图`g`是否是有向图
        '''
        dir = False
        for i in range(len(self.edges)):
            dir = dir or self.edges[i].dir != DIRECTION_NONE
        return dir

    def veterxs_atkey(self, key):
        '''
        从顶点序列`vertexs`中返回键值为`key`的顶点

        Args
        ===
        `key` Vertex | int

        '''
        if type(key) is Vertex:
            return key
        for i in range(len(g.veterxs)):
            if g.veterxs[i].key == key:
                return g.veterxs[i]

    def getvertexadj(self, v : Vertex):
        '''
        获取图中顶点`v`的邻接顶点序列
        '''
        v = self.veterxs_atkey(v)
        if v is None:
            return None
        uindex = 0
        for i in range(len(self.veterxs)):
            if self.veterxs[i].key == v.key:
                uindex = i
                break
        return self.adj[uindex]

    def getedge(self, v1 : Vertex, v2 : Vertex):
        '''
        根据两个顶点获取边，若两个点不相邻，返回None
        '''
        if type(v1) is not Vertex:
            v1 = self.veterxs_atkey(v1)
        if type(v2) is not Vertex:
            v2 = self.veterxs_atkey(v2)
        for edge in self.edges:
            if edge.vertex1.key == v1.key and edge.vertex2.key == v2.key:
                return edge
        return edge

    def printadj(self):
        '''
        打印邻接表
        '''
        for v in self.veterxs:
            list = self.getvertexadj(v)
            print(v.key, end='→')
            for e in list:
                print(e.key, end=' ')
            print('')
        
    def reset_vertex_para(self):
        '''
        复位所有顶点的参数
        '''
        for i in range(len(self.veterxs)):
            self.veterxs[i].resetpara()

    def addvertex(self, v):
        '''
        向图中添加结点`v`

        Args
        ===
        `v` : Vertex | List<Vertex> | List<string>

        '''
        if type(v) is list:
            for node in v:
                if type(node) is not Vertex:
                    key = node
                    node = Vertex(key)
                    self.veterxs.append(node)
            return
        if type(v) is not Vertex:
            key = v
            v = Vertex(key)
        self.veterxs.append(v)

    def addedgewithweight(self, v1, v2, weight, dir = DIRECTION_NONE):
        '''
        向图中添加边`edge`

        Args
        ===
        `v1` : 边的一个顶点

        `v2` : 边的另一个顶点

        `weight` : 边的权重

        `dir` : 边的方向
            DIRECTION_NONE : 没有方向
            DIRECTION_TO : `vertex1` → `vertex2`
            DIRECTION_FROM : `vertex1` ← `vertex2`
            DIRECTION_BOTH : `vertex1` ←→ `vertex2`
        '''
        egde = Edge(Vertex(v1), Vertex(v2), weight, dir)
        self.edges.append(egde)

    def addedge(self, v1, v2, dir = DIRECTION_NONE):
        '''
        向图中添加边`edge`

        Args
        ===
        `v1` : 边的一个顶点

        `v2` : 边的另一个顶点

        `dir` : 边的方向
            DIRECTION_NONE : 没有方向
            DIRECTION_TO : `vertex1` → `vertex2`
            DIRECTION_FROM : `vertex1` ← `vertex2`
            DIRECTION_BOTH : `vertex1` ←→ `vertex2`
        '''
        egde = Edge(Vertex(v1), Vertex(v2), 1, dir)
        self.edges.append(egde)

    def getvertexfromedge(self, edge : Edge):
        '''
        获取边的两个顶点的引用

        Args
        ===
        `edge` : 边 

        Return
        ===
        (u, v) : 

        '''
        n = len(self.veterxs)
        if type(edge) is Edge:
            u, v, dir = edge.vertex1, edge.vertex2, edge.dir
            for k in range(n):
                if self.veterxs[k].key == u.key:
                    uindex = k
                if self.veterxs[k].key == v.key:
                    vindex = k
            return (self.veterxs[uindex], self.veterxs[vindex])
        elif len(edge) == 2:
            u, v = edge
            uindex = self.veterxs.index(u)
            vindex = self.veterxs.index(v)
        else:
            u, v, dir = edge
            uindex = self.veterxs.index(u)
            vindex = self.veterxs.index(v)
        return (u, v)

    @property
    def adj(self):
        '''
        获取邻接表
        '''
        adj = []
        n = len(self.veterxs)
        if n == 0:
            return []
        for i in range(n):    
            sub = []     
            for edge in self.edges:
                dir = ' '
                if type(edge) is Edge:
                    u, v, dir = edge.vertex1, edge.vertex2, edge.dir
                    for k in range(n):
                        if self.veterxs[k].key == u.key:
                            uindex = k
                        if self.veterxs[k].key == v.key:
                            vindex = k
                elif len(edge) == 2:
                    u, v = edge
                    uindex = self.veterxs.index(u)
                    vindex = self.veterxs.index(v)
                else:
                    u, v, dir = edge
                    uindex = self.veterxs.index(u)
                    vindex = self.veterxs.index(v)
                if dir == DIRECTION_TO and uindex == i:
                    val = self.veterxs[vindex]
                    if sub.count(val) == 0:
                        sub.append(val)
                elif dir == DIRECTION_FROM and vindex == i:
                    val = self.veterxs[uindex]
                    if sub.count(val) == 0:
                        sub.append(val)
                elif dir == DIRECTION_NONE and uindex == i:
                    val = self.veterxs[vindex]
                    if sub.count(val) == 0:
                        sub.append(val)
                elif dir == DIRECTION_NONE and vindex == i:
                    val = self.veterxs[uindex]
                    if sub.count(val) == 0:
                        sub.append(val)               
            adj.append(sub)
        return adj

    @property
    def matrix(self):
        '''
        获取邻接矩阵,并且其是一个对称矩阵
        '''
        n = len(self.veterxs)
        if n == 0:
            return []
        mat = _np.zeros((n, n))
        for edge in self.edges:
            dir = ' '
            if type(edge) is Edge:
                u, v, dir = edge.vertex1, edge.vertex2, edge.dir 
                for k in range(n):
                    if self.veterxs[k].key == u.key:
                        uindex = k
                    if self.veterxs[k].key == v.key:
                        vindex = k
            elif len(edge) == 2:
                u, v = edge
                uindex = self.veterxs.index(u)
                vindex = self.veterxs.index(v)
            else:
                u, v, dir = edge
                uindex = self.veterxs.index(u)
                vindex = self.veterxs.index(v)                         
            if dir == DIRECTION_TO:
                mat[uindex, vindex] = 1
            elif dir == DIRECTION_FROM:
                mat[vindex, uindex] = 1
            else:
                mat[uindex, vindex] = 1
                mat[vindex, uindex] = 1
        return mat

    def gettranspose(self):
        '''
        获取图`g`的转置
        '''
        g_rev = _deepcopy(self)
        for i in range(len(g_rev.edges)):
            lastdir = g_rev.edges[i].dir
            g_rev.edges[i].dir = self.__get_rev_dir(lastdir)
        return g_rev

    def __get_rev_dir(self, dir):
        if dir == DIRECTION_FROM:
            dir = DIRECTION_TO
        elif dir == DIRECTION_TO:
            dir = DIRECTION_FROM
        else:
            dir = DIRECTION_NONE
        return dir

    def buildrevedges(self):
        '''
        构造反向的有向图边
        '''
        newedges = []
        n = len(self.edges)
        for i in range(n):
            edge = self.edges[i]
            if type(edge) is Edge:
                v1, v2, dir = edge.vertex1, edge.vertex2, edge.dir
            else:
                v1, v2, dir = edge
            edge_rev = v2, v1, self.__get_rev_dir(dir)
            newedges.append(edge_rev)
        return newedges

    def __buildBMatrix(self, B, v, i, j, v1, v2, dir):
        if v1 != v and v2 != v:
            B[i][j] = 0
        elif v1 == v and dir == DIRECTION_TO:
            B[i][j] = -1
        elif v2 == v and dir == DIRECTION_FROM:
            B[i][j] = -1
        elif v1 == v and dir == DIRECTION_FROM:
            B[i][j] = 1
        elif v2 == v and dir == DIRECTION_TO:
            B[i][j] = 1

    def buildBMatrix(self):
        '''
        构造关联矩阵
        '''
        m = len(self.veterxs)
        n = len(self.edges)
        B = _np.zeros((m, n))
        revedges = self.buildrevedges()
        for i in range(m):
            v = self.veterxs[i]
            for j in range(n):
                edge = self.edges[j]
                if type(edge) is Edge:
                    v1, v2, dir = edge.vertex1, edge.vertex2, edge.dir
                else:
                    v1, v2, dir = edge
                self.__buildBMatrix(B, v, i, j, v1, v2, dir)
            for j in range(n):
                v1, v2, dir = revedges[j]
                self.__buildBMatrix(B, v, i, j, v1, v2, dir)
        return _np.matrix(B)
    
    def contains_uni_link(self):
        '''
        确定有向图`G=(V,E)`是否包含一个通用的汇(入度为|V|-1,出度为0的点)
        '''
        n = len(self.veterxs)
        m = self.matrix
        for i in range(n):
            if sum(m[i]) == n - 1:
                return True
        return False

    @property
    def has_cycle(self):
        '''
        判断图是否有环路
        '''
        return hascircuit(self)
    
    @property
    def vertex_num(self):
        '''
        返回图中顶点数量
        '''
        return len(self.veterxs)

    @property
    def edge_num(self):
        '''
        返回图中边的数量
        '''
        return len(self.edges)

def bfs(g : Graph, s : Vertex):
    '''
    广度优先搜索(breadth-first search) 时间复杂度`O(V+E)`

    Args
    ===
    `g` : type:`Graph`,图`G(V,E)`(无向图或者有向图均可)

    `s` : type:`Vertex`，搜索的起点顶点

    Return
    ===
    None

    Example
    ===
    ```python
    from graph import *
    g = Graph()
    v = [Vertex('a'), Vertex('b'), Vertex('c'), Vertex('d'), Vertex('e')]
    g.veterxs = v
    g.edges.append(Edge(v[0], v[1]))
    g.edges.append(Edge(v[0], v[2]))
    g.edges.append(Edge(v[1], v[3]))
    g.edges.append(Edge(v[2], v[1]))
    g.edges.append(Edge(v[3], v[0]))
    g.edges.append(Edge(v[4], v[3]))
    print('邻接表为')
    print(g.adj)
    print('邻接矩阵为')
    print(g.matrix)
    for i in range(len(v)):
        bfs(g, v[i])
        print('{}到各点的距离为'.format(v[i]))
        for u in g.veterxs:
            print(u.d, end=' ')
        print(' ')
    ```
    '''
    g.reset_vertex_para()
    adj = g.adj
    # g.changeVEToClass()
    if type(s) is not Vertex:
        key = s
        for i in range(len(g.veterxs)):
            if g.veterxs[i].key == key:
                s = g.veterxs[i]
    n = len(g.veterxs)
    for i in range(n):
        u = g.veterxs[i]
        if type(u) is Vertex:
            u.color = COLOR_WHITE
            u.d = _math.inf
            u.pi = None
        else:
            return
    s.color = COLOR_GRAY
    s.d = 0
    s.pi = None
    q = []
    q.append(s)
    while len(q) != 0:
        u = q.pop(0)
        uindex = 0
        for i in range(n):
            if g.veterxs[i].key == u.key:
                uindex = i
        for i in range(len(adj[uindex])):
            v = adj[uindex][i]
            if v.color == COLOR_WHITE:
                v.color = COLOR_GRAY
                v.d = u.d + 1
                v.pi = u
                q.append(v)
        u.color = COLOR_BLACK

class _DFS:
    def __init__(self):
        self.__adj = []
        self.__sort_list = []
        self.__time = 0
        self.__n = 0
        self.__count = 0
        self.__scc_count = 0
        self.__scc_list = []

    def search_path(self, g: Graph, u: Vertex, k : Vertex):
        '''
        寻找图`g`中顶点`u`到`k`的路径
        '''
        uindex = 0
        for i in range(self.__n):
            if g.veterxs[i].key == u.key:
                uindex = i
                break   
        for i in range(len(self.__adj[uindex])):
            v = self.__adj[uindex][i]
            if v.key == k.key:
                self.__count += 1
            else:
                self.search_path(g, v, k)
        
    def dfs_visit_non_recursive(self, g: Graph, u : Vertex):
        '''
        深度优先搜索从某个顶点开始(非递归)
        '''
        stack = []
        stack.append(u)
        self.__time += 1
        u.d = self.__time
        while len(stack) > 0:
            w = stack.pop(0)
            w.color = COLOR_GRAY            
            uindex = 0
            for i in range(self.__n):
                if g.veterxs[i].key == w.key:
                    uindex = i
                    break     
            for i in range(len(self.__adj[uindex])):
                v = self.__adj[uindex][i]
                if v.color == COLOR_WHITE:
                    v.pi = w
                    stack.append(v)
                    self.__time += 1
                    v.d = self.__time
            w.color = COLOR_BLACK
            self.__time += 1
            w.f = self.__time
        u.color = COLOR_BLACK
        self.__time += 1
        u.f = self.__time

    def dfs_visit(self, g: Graph, u: Vertex):
        '''
        深度优先搜索从某个顶点开始
        '''
        u.color = COLOR_GRAY
        self.__time += 1
        u.d = self.__time
        uindex = 0
        for i in range(self.__n):
            if g.veterxs[i].key == u.key:
                uindex = i
                break
        for i in range(len(self.__adj[uindex])):
            v = self.__adj[uindex][i]
            if v.color == COLOR_WHITE:
                v.pi = u
                self.dfs_visit(g, v)
        u.color = COLOR_BLACK
        self.__time += 1
        u.f = self.__time
        self.__sort_list.append(u)

    def dfs(self, g: Graph):
        '''
        深度优先搜索算法(depth-first search) 时间复杂度`Θ(V+E)`

        Args
        ===
        `g` : type:`Graph`,图`G(V,E)`(无向图或者有向图均可)

        Return
        ===
        None

        Example
        ===
        ```python
        ```
        '''
        self.__adj = g.adj
        self.__n = len(g.veterxs)
        self.__time = 0
        self.__sort_list.clear()
        for i in range(self.__n):
            u = g.veterxs[i]
            u.color = COLOR_WHITE
            u.pi = None
        for i in range(self.__n):
            u = g.veterxs[i]
            if u.color == COLOR_WHITE:
                self.dfs_visit(g, u)
    
    def topological_sort(self, g: Graph):
        '''
        拓扑排序 时间复杂度`Θ(V+E)`

        Args
        ===
        `g` : type:`Graph`,图`G(V,E)`(无向图)

        Return
        ===
        `list` : list 排序好的顶点序列

        Example
        ===
        ```python
        import graph as _g
        g = _g.Graph()
        g.vertexs = ...
        g.edges = ...
        topological_sort(g)
        ```
        '''
        self.__sort_list.clear()
        self.dfs(g)
        sort_list = self.__sort_list
        return sort_list

    def getpathnum_betweentwovertex(self, g: Graph, v1: Vertex, v2: Vertex):
        '''
        获取有向无回路图`g`中两个顶点`v1`和`v2`之间的路径数目 时间复杂度`Θ(V+E)`
        '''
        if g.hasdirection() == False:
            print('para g 是无向图，不返回路径')
            return 0
        count = 0
        g.reset_vertex_para()
        adj = g.adj
        n = len(g.veterxs)
        if type(v1) is not Vertex:
            key = v1
            for i in range(len(g.veterxs)):
                if g.veterxs[i].key == key:
                    v1 = g.veterxs[i]
        if type(v2) is not Vertex:
            key = v2
            for i in range(len(g.veterxs)):
                if g.veterxs[i].key == key:
                    v2 = g.veterxs[i]
        self.__count = 0
        self.__adj = g.adj
        self.__n = len(g.veterxs)
        self.__time = 0
        self.search_path(g, v1, v2)
        return self.__count

    def scc(self, g : Graph):
        '''
        获取图`g`的强连通分支 时间复杂度`Θ(V+E)`
        '''
        self.__scc_count = 0
        self.__scc_list.clear()
        n = len(g.veterxs)
        g.reset_vertex_para()
        list = self.topological_sort(g)
        self.__scc_count += 1
        g_rev = g.gettranspose()
        g_rev.reset_vertex_para()
        self.dfs(g_rev)
        return self.__scc_list, self.__scc_count

__dfs_instance = _DFS()
# 深度优先搜索
dfs = __dfs_instance.dfs
# 拓扑排序
topological_sort = __dfs_instance.topological_sort
# 获得有向无环图的两个顶点间的路径个数
getpathnum_betweentwovertex = __dfs_instance.getpathnum_betweentwovertex
# 强连通分支图
scc = __dfs_instance.scc

def hascircuit_vertex(g: Graph, v : Vertex):
    '''
    判断一个无向图`g`中顶点`v`是否包含连接自己的回路 
    '''
    stack = []
    stack.append(v)
    while len(stack) > 0:      
        stack_v = stack.pop(0) 
        v_adj = g.getvertexadj(stack_v)
        stack_v.color = COLOR_GRAY
        for i in range(len(v_adj)):
            v_next = v_adj[i]
            if v_next.color == COLOR_WHITE:
                v_next.pi = stack_v
                stack.append(v_next) 
            if v_next.key == v.key and stack_v.pi is not None and stack_v.pi.key != v.key:
                return True                
        stack_v.color = COLOR_BLACK
    return False

def hascircuit(g : Graph):
    '''
    判断一个无向图`g`中是否包含回路 时间复杂度`O(V)`
    '''
    n = len(g.veterxs)
    result = False
    for i in range(n):
        v = g.veterxs[i]
        g.reset_vertex_para()
        result = result or hascircuit_vertex(g, v)
        if result == True:
            return True
    return result

def print_path(g : Graph, s : Vertex, v : Vertex):
    '''
    输出图`g`中顶点`s`到`v`的最短路径上的所有顶点

    '''
    g.reset_vertex_para()
    bfs(g, s)
    if type(s) is not Vertex:
        key = s
        for i in range(len(g.veterxs)):
            if g.veterxs[i].key == key:
                s = g.veterxs[i]
    if type(v) is not Vertex:
        key = v
        for i in range(len(g.veterxs)):
            if g.veterxs[i].key == key:
                v = g.veterxs[i]
    if v == s:
        print('{}→'.format(s.key), end='')
    elif v.pi == None:
        print('no path from {} to {} exists'.format(s.key, v.key))
    else:
        print_path(g, s, v.pi)
        print('{}→'.format(v.key), end='')

def undirected_graph_test():
    '''
    测试无向图
    '''
    g = Graph()
    g.veterxs = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    g.edges = [('a', 'b'), ('a', 'c'), ('b', 'd'),
               ('b', 'e'), ('c', 'f'), ('c', 'g')]
    print('邻接表为')
    print(g.adj)
    print('邻接矩阵为')
    print(g.matrix)

def directed_graph_test():
    '''
    测试有向图
    '''
    g = Graph()
    g.veterxs = ['1', '2', '3', '4', '5', '6']
    g.edges = [('1', '2', '→'), ('4', '2', '→'), 
               ('1', '4', '→'), ('2', '5', '→'),
               ('3', '6', '→'), ('3', '5', '→'),
               ('5', '4', '→'), ('6', '6', '→')]
    print('邻接表为')
    print(g.adj)
    print('邻接矩阵为')
    print(g.matrix)
    B = g.buildBMatrix()
    print('关联矩阵为')
    print(B)
    print(B * B.T)
    print('是否包含通用的汇', g.contains_uni_link())

def test_bfs():
    '''
    测试广度优先搜索方法
    '''
    g = Graph()
    v = [Vertex('a'), Vertex('b'), Vertex('c'), Vertex('d'), Vertex('e')]
    g.veterxs = v
    g.edges.clear()
    g.edges.append(Edge(v[0], v[1]))
    g.edges.append(Edge(v[0], v[2]))
    g.edges.append(Edge(v[1], v[3]))
    g.edges.append(Edge(v[2], v[1]))
    g.edges.append(Edge(v[3], v[0]))
    g.edges.append(Edge(v[4], v[3]))
    print('邻接表为')
    print(g.adj)
    print('邻接矩阵为')
    print(g.matrix)
    for i in range(len(v)):
        bfs(g, v[i])
        print('{}到各点的距离为'.format(v[i]))
        for u in g.veterxs:
            print(u.d, end=' ')
        print(' ')
    bfs(g, v[0])
    print_path(g, v[0], v[4])
    print('')
    del g

    gwithdir = Graph()
    vwithdir = [Vertex('a'), Vertex('b'), Vertex('c'),
                Vertex('d'), Vertex('e')]
    gwithdir.veterxs = vwithdir
    gwithdir.edges.clear()
    gwithdir.edges.append(Edge(vwithdir[0], vwithdir[1], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[1], vwithdir[2], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[2], vwithdir[3], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[3], vwithdir[4], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[0], vwithdir[2], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[2], vwithdir[4], 1, DIRECTION_TO))
    print('邻接表为')
    print(gwithdir.adj)
    print('邻接矩阵为')
    print(gwithdir.matrix)
    for i in range(len(vwithdir)):
        bfs(gwithdir, vwithdir[i])
        print('{}到各点的距离为'.format(vwithdir[i]))
        for u in gwithdir.veterxs:
            print(u.d, end=' ')
        print('')
    bfs(gwithdir, vwithdir[0])
    print_path(gwithdir, vwithdir[0], vwithdir[4])
    print('')
    del gwithdir

def test_dfs():
    '''
    测试深度优先搜索方法
    '''
    gwithdir = Graph()
    vwithdir = [Vertex('a'), Vertex('b'), Vertex('c'),
                Vertex('d'), Vertex('e')]
    gwithdir.veterxs = vwithdir
    gwithdir.edges.clear()
    gwithdir.edges.append(Edge(vwithdir[0], vwithdir[1], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[0], vwithdir[2], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[1], vwithdir[3], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[2], vwithdir[1], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[3], vwithdir[0], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[4], vwithdir[3], 1, DIRECTION_FROM))
    print('邻接表为')
    print(gwithdir.adj)
    print('邻接矩阵为')
    print(gwithdir.matrix)
    dfs(gwithdir)
    print('')
    del gwithdir

def _print_inner_conllection(collection : list, end='\n'):
    '''
    打印列表内部内容
    '''
    print('[',end=end)
    for i in range(len(collection)):
        if type(collection[i]) is list: 
            _print_inner_conllection(collection[i], end)
        else:
            print(str(collection[i]), end=end)
    print(']')

def test_topological_sort():
    '''
    测试拓扑排序
    '''
    gwithdir = Graph()
    vwithdir = [Vertex('a'), Vertex('b'), Vertex('c'),
                Vertex('d'), Vertex('e')]
    gwithdir.veterxs = vwithdir
    gwithdir.edges.clear()
    gwithdir.edges.append(Edge(vwithdir[0], vwithdir[1], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[1], vwithdir[2], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[2], vwithdir[3], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[3], vwithdir[4], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[0], vwithdir[2], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[2], vwithdir[4], 1, DIRECTION_TO))
    print('邻接表为')
    print(gwithdir.adj)
    print('邻接矩阵为')
    print(gwithdir.matrix)
    sort_list = topological_sort(gwithdir)
    _print_inner_conllection(sort_list)
    print('')
    print('a到e的路径个数为：', getpathnum_betweentwovertex(gwithdir, 'a', 'e'))

def test_hascircuit():
    '''
    测试是否包含环路函数
    '''
    gwithdir = Graph()
    vwithdir = [Vertex('a'), Vertex('b'), Vertex('c'),
                Vertex('d'), Vertex('e')]
    gwithdir.veterxs = vwithdir
    gwithdir.edges.clear()
    gwithdir.edges.append(Edge(vwithdir[0], vwithdir[1]))
    gwithdir.edges.append(Edge(vwithdir[1], vwithdir[2]))
    gwithdir.edges.append(Edge(vwithdir[2], vwithdir[3]))
    gwithdir.edges.append(Edge(vwithdir[3], vwithdir[4]))
    print('是否包含环路：', hascircuit(gwithdir))
    gwithdir.edges.append(Edge(vwithdir[0], vwithdir[2]))
    gwithdir.edges.append(Edge(vwithdir[2], vwithdir[4]))
    print('是否包含环路：', hascircuit(gwithdir))

def test_scc():
    '''
    测量强连通分支算法
    '''
    g = Graph()
    g.veterxs.clear()
    g.edges.clear()
    v = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    g.addvertex(v)
    g.addedge('a', 'b', DIRECTION_TO)
    g.addedge('b', 'c', DIRECTION_TO)
    g.addedge('c', 'd', DIRECTION_TO)
    g.addedge('d', 'c', DIRECTION_TO)
    g.addedge('e', 'a', DIRECTION_TO)
    g.addedge('b', 'e', DIRECTION_TO)
    g.addedge('b', 'f', DIRECTION_TO)
    g.addedge('c', 'g', DIRECTION_TO)
    g.addedge('d', 'h', DIRECTION_TO)
    g.addedge('h', 'd', DIRECTION_TO)
    g.addedge('e', 'f', DIRECTION_TO)
    g.addedge('f', 'g', DIRECTION_TO)
    g.addedge('g', 'f', DIRECTION_TO)
    g.addedge('h', 'g', DIRECTION_TO)
    scc(g)

def test():
    '''
    测试函数
    '''
    undirected_graph_test()
    directed_graph_test()
    test_bfs()
    test_dfs()
    test_topological_sort()
    test_hascircuit()
    test_scc()

if __name__ == '__main__':
    test()
else:
    pass

```

[Github Code](https://github.com/Peefy/IntroductionToAlgorithm.Python/blob/master/src/chapter22)
