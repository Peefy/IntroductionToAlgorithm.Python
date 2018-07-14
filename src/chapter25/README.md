---
layout: post
title: "用Python实现的单源最短路径算法
description: "用Python实现的单源最短路径算法"
categories: [Python]
tags: [python]
redirect_from:
  - /2018/07/13
---


## 单源最短路径

在最短路径问题中,给出的是一个带权有向图G=(V,E),加权函数w:E->R为从边到实型权值的映射

路径p=<v0,v1,...,vk>的权是指其组成边的所有权值之和

广度优先搜索算法就是一种在无权图上执行的最短路径算法

```python

import graph as _g
import math as _math
from copy import deepcopy as _deepcopy

class _ShortestPath:
    '''
    单源最短路径算法集合
    '''
    def __init__(self, *args, **kwords):
        pass

    def initialize_single_source(self, g : _g.Graph, s : _g.Vertex):
        '''
        最短路径估计和前趋进行初始化 时间复杂度Θ(V)
        '''
        for v in g.veterxs:
            v.d = _math.inf
            v.pi = None
        s.d = 0

    def relax(self, u : _g.Vertex, v : _g.Vertex, weight):
        '''
        一步松弛操作
        '''
        if v.d > u.d + weight:
            v.d = u.d + weight
            v.pi = u
    
    def bellman_ford(self, g : _g.Graph, s : _g.Vertex):
        '''
        Bellmax-Ford算法能在一般的情况下(存在负边权的情况)下,解决单源最短路径问题

        时间复杂度 O(VE)

        Args
        ===
        `g` : 图G=(V,E)

        `w` : 加权函数

        `s` : 源顶点

        Return
        ===
        `exist` : bool 返回一个布尔值,表明图中是否存在着一个从源点可达的权为负的回路
        若存在这样的回路的话,算法说明该问题无解;若不存在这样的回路,算法将产生最短路径以及权值

        '''
        if type(s) is not _g.Vertex:
            s = g.veterxs_atkey(s)
        self.initialize_single_source(g, s)
        n = g.vertex_num
        for i in range(n - 1):
            for edge in g.edges:
                u, v = edge.vertex1, edge.vertex2
                u = g.veterxs_atkey(u)
                v = g.veterxs_atkey(v)
                self.relax(u, v, edge.weight)
        for edge in g.edges:
            u, v = edge.vertex1, edge.vertex2
            u = g.veterxs_atkey(u)
            v = g.veterxs_atkey(v)
            if v.d > u.d + edge.weight:
                return False
        return True 
    
    def dag_shortest_path(self, g : _g.Graph, s : _g.Vertex):
        '''
        按顶点的拓扑序列对某加权dag图(有向无回路图)G=(V,E)的边进行松弛后
        就可以在Θ(V+E)时间内计算出单源最短路径.

        Args
        ===
        `g` : 有向无回路图G=(V,E) 

        `s` : 源顶点

        '''
        sort_list = _g.topological_sort(g)
        self.initialize_single_source(g, s)
        for u in sort_list:
            u = g.veterxs_atkey(u)
            adj = g.getvertexadj(u)
            for v in adj:
                edge = g.getedge(u, v)
                self.relax(u, v, edge.weight)
            
    def dijstra(self, g : _g.Graph, s : _g.Vertex):
        '''
        单源最短路径Dijstra算法
        '''
        self.initialize_single_source(g, s)
        S = []
        Q = g.veterxs
        while len(Q) != 0:
            Q.sort(reverse=True)
            u = Q.pop()
            S += [u]
            adj = g.getvertexadj(u)
            if adj is not None:
                for v in adj:
                    edge = g.getedge(u, v)
                    self.relax(u, v, edge.weight)

__shortest_path_instance = _ShortestPath()
bellman_ford = __shortest_path_instance.bellman_ford
dag_shortest_path = __shortest_path_instance.dag_shortest_path
dijstra = __shortest_path_instance.dijstra

def test_bellman_ford():
    g = _g.Graph()
    g.clear()
    vertexs = [_g.Vertex('s'), _g.Vertex('t'), _g.Vertex(
        'x'), _g.Vertex('y'), _g.Vertex('z')]
    g.veterxs = vertexs
    g.addedgewithweight('s', 't', 6, _g.DIRECTION_TO)
    g.addedgewithweight('s', 'y', 7, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'x', 5, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'y', 8, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'z', -4, _g.DIRECTION_TO)
    g.addedgewithweight('x', 't', -2, _g.DIRECTION_TO)
    g.addedgewithweight('y', 'x', -3, _g.DIRECTION_TO)
    g.addedgewithweight('y', 'z', 9, _g.DIRECTION_TO)
    g.addedgewithweight('z', 'x', 7, _g.DIRECTION_TO)
    g.addedgewithweight('z', 's', 2, _g.DIRECTION_TO)
    print(bellman_ford(g, vertexs[0]))
    del g

def test_dag_shortest_path():
    g = _g.Graph()
    g.clear()
    vertexs = [_g.Vertex('r'), _g.Vertex('s'), _g.Vertex('t'),
        _g.Vertex('x'), _g.Vertex('y'), _g.Vertex('z')]
    g.veterxs = vertexs
    g.addedgewithweight('r', 's', 5, _g.DIRECTION_TO)
    g.addedgewithweight('s', 't', 2, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'x', 7, _g.DIRECTION_TO)
    g.addedgewithweight('x', 'y', -1, _g.DIRECTION_TO)
    g.addedgewithweight('y', 'z', -2, _g.DIRECTION_TO)
    g.addedgewithweight('r', 't', 3, _g.DIRECTION_TO)
    g.addedgewithweight('s', 'x', 6, _g.DIRECTION_TO)
    g.addedgewithweight('x', 'z', 1, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'y', 4, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'z', 2, _g.DIRECTION_TO)
    g.reset_vertex_para()
    dag_shortest_path(g, vertexs[0])
    del g

def test_dijstra():
    g = _g.Graph()
    g.clear()
    vertexs = [_g.Vertex('r'), _g.Vertex('s'), _g.Vertex('t'),
        _g.Vertex('x'), _g.Vertex('y'), _g.Vertex('z')]
    g.veterxs = vertexs
    g.addedgewithweight('r', 's', 5, _g.DIRECTION_TO)
    g.addedgewithweight('s', 't', 2, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'x', 7, _g.DIRECTION_TO)
    g.addedgewithweight('x', 'y', -1, _g.DIRECTION_TO)
    g.addedgewithweight('y', 'z', -2, _g.DIRECTION_TO)
    g.addedgewithweight('r', 't', 3, _g.DIRECTION_TO)
    g.addedgewithweight('s', 'x', 6, _g.DIRECTION_TO)
    g.addedgewithweight('x', 'z', 1, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'y', 4, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'z', 2, _g.DIRECTION_TO)
    g.reset_vertex_para()
    dijstra(g, vertexs[0])
    del g

def test():
    '''
    测试函数
    '''
    test_bellman_ford()
    test_dag_shortest_path()
    test_dijstra()

if __name__ == '__main__':
    test()
else:
    pass

```

[Github Code](https://github.com/Peefy/IntroductionToAlgorithm.Python/blob/master/src/chapter24)
