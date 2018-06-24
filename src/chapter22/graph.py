
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

    def __str__(self):
        return '[key:{} color:{} d:{} pi:{}]'.format(self.key, self.color, self.d, self.pi)

class Edge:
    '''
    图的边，包含两个顶点
    '''
    def __init__(self, vertex1 : Vertex = None, \
            vertex2 : Vertex = None, \
            distance = 1, \
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
        self.distance = distance

    def __str__(self):
        return str((self.vertex1, self.vertex2, self.dir))

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
        self.adj = []
        self.matrix = []
   
    def addvertex(self, v):
        '''
        向图中添加结点`v`
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

    def getadj(self):
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
                    sub.append(self.veterxs[vindex])
                elif dir == DIRECTION_FROM and vindex == i:
                    sub.append(self.veterxs[uindex])
                elif dir == DIRECTION_NONE and uindex == i:
                    sub.append(self.veterxs[vindex])
                # elif dir == DIRECTION_NONE and vindex == i:
                #    sub.append(self.veterxs[uindex])               
            adj.append(sub)
        self.adj = adj
        return adj

    def getmatrix(self):
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
        self.matrix = mat
        return mat

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
        self.getmatrix()
        m = self.matrix
        for i in range(n):
            if sum(m[i]) == n - 1:
                return True
        return False

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
    print(g.getadj())
    print('邻接矩阵为')
    print(g.getmatrix())
    for i in range(len(v)):
        bfs(g, v[i])
        print('{}到各点的距离为'.format(v[i]))
        for u in g.veterxs:
            print(u.d, end=' ')
        print(' ')
    ```
    '''
    adj = g.getadj()
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
        self.__time = 0
        self.__n = 0
        self.__adj = []

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
        self.__adj = g.getadj()
        self.__n = len(g.veterxs)
        self.__time = 0
        for i in range(self.__n):
            u = g.veterxs[i]
            u.color = COLOR_WHITE
            u.pi = None
        for i in range(self.__n):
            u = g.veterxs[i]
            if u.color == COLOR_WHITE:
                # self.dfs_visit_non_recursive(g, u)
                self.dfs_visit(g, u)

__dfs_instance = _DFS()
dfs = __dfs_instance.dfs
          
def print_path(g : Graph, s : Vertex, v : Vertex):
    '''
    输出图`g`中顶点`s`到`v`的最短路径上的所有顶点

    假设已经运行了BFS来计算最短路径
    '''
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
    print(g.getadj())
    print('邻接矩阵为')
    print(g.getmatrix())

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
    print(g.getadj())
    print('邻接矩阵为')
    print(g.getmatrix())
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
    print(g.getadj())
    print('邻接矩阵为')
    print(g.getmatrix())
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
    gwithdir.edges.append(Edge(vwithdir[0], vwithdir[2], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[1], vwithdir[3], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[2], vwithdir[1], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[3], vwithdir[0], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[4], vwithdir[3], 1, DIRECTION_TO))
    print('邻接表为')
    print(gwithdir.getadj())
    print('邻接矩阵为')
    print(gwithdir.getmatrix())
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
    print(gwithdir.getadj())
    print('邻接矩阵为')
    print(gwithdir.getmatrix())
    dfs(gwithdir)
    print('')

def test():
    undirected_graph_test()
    directed_graph_test()
    test_bfs()
    test_dfs()

if __name__ == '__main__':
    test()
else:
    pass
