

## 图的基本算法

基于广度优先或深度优先图搜索的算法

## 图的表示

要表示一个有向图或者无向图G=(V,E),有两种标准的方法，即邻接表和邻接矩阵
这两种表示法即可以用于有向图，也可以用于无向图
通常采用邻接表表示法，因为用这种方法表示稀疏图比较紧凑

## 广度优先搜索

广度优先搜素是最简单的图搜索算法之一,也是很多重要的图算法的原型

```python

import math as _math

from numpy import *

class Graph:
    '''
    图`G=(V,E)`
    '''
    def __init__(self, vertexs : list = None, edges : list = None):
        '''
        图`G=(V,E)`

        Args
        ===
        `vertexs` : 图的顶点

        `edges` : 图的边

        '''
        self.veterxs = vertexs
        self.edges = edges
   
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
                if len(edge) == 2:
                    u, v = edge
                else:
                    u, v, dir = edge
                uindex = self.veterxs.index(u)
                vindex = self.veterxs.index(v)
                if dir == '→' and uindex == i:
                    sub.append(v)
                elif uindex == i:
                    sub.append(v)
            adj.append(sub)
        return adj

    def getmatrix(self):
        '''
        获取邻接矩阵,并且其是一个对称矩阵
        '''
        n = len(self.veterxs)
        if n == 0:
            return []
        mat = zeros((n, n))
        for edge in self.edges:
            dir = ' '
            if len(edge) == 2:
                u, v = edge
            else:
                u, v, dir = edge
            uindex = self.veterxs.index(u)
            vindex = self.veterxs.index(v)                         
            if dir == '→':
                mat[uindex, vindex] = 1
            elif dir == '←':
                mat[vindex, uindex] = 1
            else:
                mat[uindex, vindex] = 1
                mat[vindex, uindex] = 1
        return mat

    def __get_rev_dir(self, dir):
        if dir == '→':
            dir = '←'
        elif dir == '←':
            dir = '→'
        else:
            dir = ' '
        return dir

    def buildrevedges(self):
        '''
        构造反向的有向图边
        '''
        newedges = []
        n = len(self.edges)
        for i in range(n):
            edge = self.edges[i]
            v1, v2, dir = edge
            edge_rev = v2, v1, self.__get_rev_dir(dir)
            newedges.append(edge_rev)
        return newedges

    def buildBMatrix(self):
        '''
        构造关联矩阵
        '''
        m = len(self.veterxs)
        n = len(self.edges)
        B = zeros((m, n))
        revedges = self.buildrevedges()
        for i in range(m):
            v = self.veterxs[i]
            for j in range(n):
                v1, v2, dir = self.edges[j]
                if v1 != v and v2 != v:
                    B[i][j] = 0
                elif v1 == v and dir == '→':
                    B[i][j] = -1
                elif v2 == v and dir == '←':
                    B[i][j] = -1
                elif v1 == v and dir == '←':
                    B[i][j] = 1
                elif v2 == v and dir == '→ ':
                    B[i][j] = 1
            for j in range(n):
                v1, v2, dir = revedges[j]
                if v1 != v and v2 != v:
                    B[i][j] = 0
                elif v1 == v and dir == '→':
                    B[i][j] = -1
                elif v2 == v and dir == '←':
                    B[i][j] = -1
                elif v1 == v and dir == '←':
                    B[i][j] = 1
                elif v2 == v and dir == '→ ':
                    B[i][j] = 1
        return matrix(B)
    
    def contains_uni_link(self):
        '''
        确定有向图`G=(V,E)`是否包含一个通用的汇(入度为|V|-1,出度为0的点)
        '''
        return False

def undirected_graph_test():
    g = Graph()
    g.veterxs = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    g.edges = [('a', 'b'), ('a', 'c'), ('b', 'd'),
               ('b', 'e'), ('c', 'f'), ('c', 'g')]
    print('邻接表为')
    print(g.getadj())
    print('邻接矩阵为')
    print(g.getmatrix())

def directed_graph_test():
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

def test():
    undirected_graph_test()
    directed_graph_test()

if __name__ == '__main__':
    test()
else:
    pass
```

[Github Code](https://github.com/Peefy/CLRS_dugu_code-master/blob/master/src/chapter22)
