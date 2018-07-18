---
layout: post
title: "用Python实现的每对顶点间的最短路径算法
description: "用Python实现的每对顶点间的最短路径算法"
categories: [Python]
tags: [python]
redirect_from:
  - /2018/07/18
---


## 每对顶点间的最短路径

```python
import math as _math
from copy import deepcopy as _deepcopy
import numpy as _np

import graph as _g
import shortestpath as _sp

class _ExtendShortestPath:
    '''
    扩展最短路径对算法集合类
    '''
    def __init__(self, *args, **kwords):
        '''
        扩展最短路径对算法集合类
        '''
        pass

    def extend_shortest_paths(self, L, W):
        '''
        最短路径对矩阵`L`单步迭代过程
        '''
        n = _np.shape(L)[0] # rows of L
        L_return = _np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                L_return[i][j] = _math.inf
                for k in range(n):
                    L_return[i][j] = min(L_return[i][j], L[i][k] + W[k][j])
        return L_return

    def show_all_pairs_shortest_paths(self, W):
        '''
        根据图`g`的权值矩阵`W`求所有最短路径对矩阵
        '''
        n = _np.shape(W)[0] # rows of W
        L = list(range(n))
        L[0] = W
        for m in range(1, n - 1):
            L[m] = self.extend_shortest_paths(L[m - 1], W)
        return L[n - 2]

    def faster_all_pairs_shortest_paths(self, W):
        '''
        根据图`g`的权值矩阵`W`求所有最短路径对矩阵
        '''
        n = _np.shape(W)[0] # rows of W
        L_last = W
        L_now = []
        m = 1
        while m < n - 1:
            L_now = self.extend_shortest_paths(L_last, L_last)
            m = 2 * m
            L_last = L_now
        return L_now

    def getpimatrix(self, g, L, W):
        '''
        获得前趋矩阵`∏`
        '''
        n = _np.shape(W)[0] # rows of W
        pi = _np.zeros((n, n), dtype=_np.str)
        index = 0
        for i in range(n):
            for j in range(n):
                pi[i][j] = '∞'
                if i == j:
                    pi[i][j] = g.veterxs[i].key
                else:
                    if L[i][j] == _math.inf:
                        pi[i][j] = '∞'
                        continue
                    if L[i][j] == W[i][j]:
                        pi[i][j] = g.veterxs[j].key
                        continue
                    for k in range(n):
                        if k != i and k !=j and L[i][j] == L[i][k] + L[k][j]:
                            pi[i][j] = g.veterxs[k].key
        return pi

    def floyd_warshall_step(self, D_last, k):
        '''
        单步`Floyd-Warshall`算法
        '''
        n = _np.shape(D_last)[0] # rows of W
        D = _deepcopy(D_last)
        for i in range(n):
            for j in range(n):
                D[i][j] = min(D_last[i][j], D_last[i][k] + D_last[k][j])
        return D

    def floyd_warshall(self, W, pi):
        '''
        根据图`g`的权值矩阵`W`求最短路径对矩阵的`Floyd-Warshall`算法
        '''
        n = _np.shape(W)[0] # rows of W
        D = W
        D_last = W
        P = pi
        P_last = pi
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if D_last[i][j] > D_last[i][k] + D_last[k][j]:
                        D[i][j] = D_last[i][k] + D_last[k][j]
                        P[i][j] = P_last[k][j]
                    else:
                        D[i][j] = D_last[i][j]
                        P[i][j] = P_last[i][j]
            D_last = D
            P_last = P
        return D, P

    def transitive_closure(self, g : _g.Graph):
        '''
        有向图`g`的传递闭包
        '''
        n = g.vertex_num
        t = _np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                edge = g.getedge(g.veterxs[i], g.veterxs[j])
                if i == j or edge in g.edges:
                    t[i][j] = 1
                else:
                    t[i][j] = 0
        t_last = _deepcopy(t)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    result = t_last[i][j] or (t_last[i][k] and t_last[k][j])
                    if result == True:
                        t[i][j] = 1
                    else:
                        t[i][j] = 0
            t_last = t
        return t

    def johnson(self, g : _g.Graph):
        '''
        根据图`g`的邻接表`adj`求最短路径对矩阵的`Johnson`算法
        '''
        new_g = _deepcopy(g)
        s = _g.Vertex('0')
        # V[G`] = V[G]∪{s}
        new_g.addvertex(s)
        # E[G`] = E[G]∪{(s,v),v∈V}
        # w(s, v) = 0 for all v∈V[G]
        for v in new_g.veterxs:
            new_g.addedgewithdir(s, v, 0)
        exist, weight = _sp.bellman_ford(new_g, s)
        if exist == False:
            print('the graph contains a negative-weight cycle')
        else:
            n = new_g.vertex_num
            D = _np.zeros((n, n))
            for v in new_g.veterxs:
                v.d = weight
            for edge in new_g.edges:
                u, v = new_g.getvertexfromedge(edge)
                edge.weight = edge.weight + u.d - v.d
            for u in new_g.veterxs:
                _sp.dijstra(new_g, u)
                uindex = new_g.getvertexindex(u)
                for v in new_g.veterxs:
                    vindex = new_g.getvertexindex(v)
                    edge = new_g.getedge(u, v)
                    if edge is not None:
                        D[uindex][vindex] = edge.weight + v.d - u.d
            return D
        
__esp_instance = _ExtendShortestPath()

extend_shortest_paths = __esp_instance.extend_shortest_paths
show_all_pairs_shortest_paths = __esp_instance.show_all_pairs_shortest_paths
faster_all_pairs_shortest_paths = __esp_instance.faster_all_pairs_shortest_paths
getpimatrix = __esp_instance.getpimatrix
floyd_warshall_step = __esp_instance.floyd_warshall_step 
floyd_warshall = __esp_instance.floyd_warshall
transitive_closure = __esp_instance.transitive_closure
johnson = __esp_instance.johnson

def test_show_all_pairs_shortest_paths():
    g = _g.Graph()
    vertexs = ['1', '2', '3', '4', '5']
    g.addvertex(vertexs)
    g.addedgewithweight('1', '2', 3, _g.DIRECTION_TO)
    g.addedgewithweight('1', '3', 8, _g.DIRECTION_TO)
    g.addedgewithweight('1', '5', -4, _g.DIRECTION_TO)
    g.addedgewithweight('2', '4', 1, _g.DIRECTION_TO)
    g.addedgewithweight('2', '5', 7, _g.DIRECTION_TO)
    g.addedgewithweight('3', '2', 4, _g.DIRECTION_TO)
    g.addedgewithweight('4', '1', 2, _g.DIRECTION_TO)
    g.addedgewithweight('4', '3', -5, _g.DIRECTION_TO)
    g.addedgewithweight('5', '4', 6, _g.DIRECTION_TO)
    W = g.getmatrixwithweight()
    print('带权值的邻接矩阵为：')
    print(W)
    print('显示所有的最短路径')
    print(show_all_pairs_shortest_paths(W))
    print('显示所有的最短路径(对数加速)')
    L = faster_all_pairs_shortest_paths(W)
    print(L)
    print('pi矩阵为')
    pi = getpimatrix(g, L, W)
    print(pi)

def test_floyd_warshall():
    g = _g.Graph()
    vertexs = ['1', '2', '3', '4', '5']
    g.addvertex(vertexs)
    g.addedgewithweight('1', '2', 3, _g.DIRECTION_TO)
    g.addedgewithweight('1', '3', 8, _g.DIRECTION_TO)
    g.addedgewithweight('1', '5', -4, _g.DIRECTION_TO)
    g.addedgewithweight('2', '4', 1, _g.DIRECTION_TO)
    g.addedgewithweight('2', '5', 7, _g.DIRECTION_TO)
    g.addedgewithweight('3', '2', 4, _g.DIRECTION_TO)
    g.addedgewithweight('4', '1', 2, _g.DIRECTION_TO)
    g.addedgewithweight('4', '3', -5, _g.DIRECTION_TO)
    g.addedgewithweight('5', '4', 6, _g.DIRECTION_TO)
    W = g.getmatrixwithweight()
    print('带权值的邻接矩阵为：')
    print(W)
    print('初始前趋矩阵')
    pi = g.getpimatrix()
    print(pi)
    print('所有的最短路径')
    D, P = floyd_warshall(W, pi)
    print(D)
    print('最终前趋路径')
    print(P)

def test_transitive_closure():
    g = _g.Graph()
    vertexs = ['1', '2', '3', '4']
    g.addvertex(vertexs)
    g.addedge('4', '1', _g.DIRECTION_TO)
    g.addedge('4', '3', _g.DIRECTION_TO)
    g.addedge('2', '4', _g.DIRECTION_TO)
    g.addedge('2', '3', _g.DIRECTION_TO)
    g.addedge('3', '2', _g.DIRECTION_TO)
    mat = g.getmatrix()
    print('邻接矩阵')
    print(mat)
    t = transitive_closure(g)
    print('传递闭包')
    print(t)

def test_johnson():
    g = _g.Graph()
    vertexs = ['1', '2', '3', '4']
    g.addvertex(vertexs)
    g.addedge('4', '1', _g.DIRECTION_TO)
    g.addedge('4', '3', _g.DIRECTION_TO)
    g.addedge('2', '4', _g.DIRECTION_TO)
    g.addedge('2', '3', _g.DIRECTION_TO)
    g.addedge('3', '2', _g.DIRECTION_TO)
    mat = g.getmatrix()
    print('邻接矩阵')
    print(mat)
    print(johnson(g))

def test():
    test_show_all_pairs_shortest_paths()
    test_floyd_warshall()
    test_transitive_closure()
    test_johnson()

if __name__ == '__main__':
    test()
else:
    pass

```

[Github Code](https://github.com/Peefy/IntroductionToAlgorithm.Python/blob/master/src/chapter25)
