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

import graph as _g
import math as _math
from copy import deepcopy as _deepcopy
from numpy import *

class _ExtendShortestPath:
    def __init__(self, *args, **kwords):
        pass

    def extend_shortest_paths(self, L, W):
        n = shape(L)[0] # rows of L
        L_return = zeros((n, n))
        for i in range(n):
            for j in range(n):
                L_return[i][j] = _math.inf
                for k in range(n):
                    L_return[i][j] = min(L_return[i][j], L[i][k] + W[k][j])
        return L_return

    def show_all_pairs_shortest_paths(self, W):
        n = shape(W)[0] # rows of W
        L = list(range(n))
        L[0] = W
        for m in range(1, n - 1):
            L[m] = self.extend_shortest_paths(L[m - 1], W)
        return L[n - 2]

__esp_instance = _ExtendShortestPath()
extend_shortest_paths = __esp_instance.extend_shortest_paths
show_all_pairs_shortest_paths = __esp_instance.show_all_pairs_shortest_paths

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

def test():
    test_show_all_pairs_shortest_paths()

if __name__ == '__main__':
    test()
else:
    pass

```

[Github Code](https://github.com/Peefy/IntroductionToAlgorithm.Python/blob/master/src/chapter25)
