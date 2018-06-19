
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
        self.adj = []

    def getmatrix(self):
        '''
        获取邻接矩阵,并且其是一个对称矩阵
        '''
        n = len(self.veterxs)
        if n == 0:
            return []
        mat = zeros((n, n))
        for edge in self.edges:
            u, v = edge
            uindex = self.veterxs.index(u)
            vindex = self.veterxs.index(v)
            mat[uindex, vindex] = 1
            mat[vindex, uindex] = 1
        return mat

def test():
    g = Graph()
    g.veterxs = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    g.edges = [('a', 'b'), ('a', 'c'), ('b', 'd'),
               ('b', 'e'), ('c', 'f'), ('c', 'g')]
    print(g.getmatrix())

if __name__ == '__main__':
    test()
else:
    pass
