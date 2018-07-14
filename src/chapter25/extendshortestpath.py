
import graph as _g
import math as _math
from copy import deepcopy as _deepcopy
from numpy import *
import numpy as np

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

    def faster_all_pairs_shortest_paths(self, W):
        n = shape(W)[0] # rows of W
        L_last = W
        L_now = []
        m = 1
        while m < n - 1:
            L_now = self.extend_shortest_paths(L_last, L_last)
            m = 2 * m
            L_last = L_now
        return L_now

    def getpimatrix(self, g, L, W):
        n = shape(W)[0] # rows of W
        pi = zeros((n, n), dtype=np.str)
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

__esp_instance = _ExtendShortestPath()
extend_shortest_paths = __esp_instance.extend_shortest_paths
show_all_pairs_shortest_paths = __esp_instance.show_all_pairs_shortest_paths
faster_all_pairs_shortest_paths = __esp_instance.faster_all_pairs_shortest_paths
getpimatrix = __esp_instance.getpimatrix

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

def test():
    test_show_all_pairs_shortest_paths()

if __name__ == '__main__':
    test()
else:
    pass

