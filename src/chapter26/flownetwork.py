
import math as _math
from copy import deepcopy as _deepcopy

from graph import * 

class _FlowNetwork:
    '''
    流网络相关算法集合类
    '''
    def __init__(self, *args, **kwargs):
        '''
        流网络相关算法集合类
        '''
        pass

    def _findbfs(self):
        '''
        广度搜索算法寻找是否存在增广路径`p`
        '''
        return False

    def ford_fulkerson(self, g : Graph, s : Vertex, t : Vertex):
        '''
        基本的Ford-Fulkerson算法
        '''
        for edge in g.edges:
            edge.flowtofrom = 0
            edge.flowfromto = 0
    
    def edmonds_karp(self, g : Graph, s : Vertex, t : Vertex):
        '''
        使用广度优先搜索实现增广路径`p`计算的Edmonds-Karp算法
        '''
        for edge in g.edges:
            edge.flowtofrom = 0
            edge.flowfromto = 0

__fn_instance = _FlowNetwork()

ford_fulkerson = __fn_instance.ford_fulkerson
edmonds_karp = __fn_instance.edmonds_karp

def _buildtestgraph():
    '''
    构造测试有向图`G=(V,E)`
    '''
    g = Graph()
    vertexs = ['s', 't', 'v1', 'v2', 'v3', 'v4']
    g.addvertex(vertexs)
    g.addedgewithdir('s', 'v1', 16)
    g.addedgewithdir('s', 'v2', 13)
    g.addedgewithdir('v1', 'v2', 10)
    g.addedgewithdir('v2', 'v1', 4)
    g.addedgewithdir('v1', 'v3', 12)
    g.addedgewithdir('v2', 'v4', 14)
    g.addedgewithdir('v3', 'v2', 9)
    g.addedgewithdir('v3', 't', 20)
    g.addedgewithdir('v4', 't', 4)
    g.addedgewithdir('v4', 'v3', 7)
    return g

def test_ford_fulkerson():
    '''
    测试基本的Ford-Fulkerson算法
    '''
    g = _buildtestgraph()
    print('邻接矩阵为')
    print(g.getmatrixwithweight())

def test_edmonds_karp():
    '''
    测试Edmonds-Karp算法
    '''
    g = _buildtestgraph()
    print('邻接矩阵为')
    print(g.getmatrixwithweight())

def test():
    '''
    测试函数
    '''
    test_ford_fulkerson()
    test_edmonds_karp()

if __name__ == '__main__':
    test()
else:
    pass
