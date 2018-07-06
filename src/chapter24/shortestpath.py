
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