
from graph import *
from copy import deepcopy

def approx_vertex_cover(g : Graph):
    """
    顶点覆盖问题的近似算法
    """
    C = []
    edges = deepcopy(g.edges)
    while len(edges) != 0:
        u, v = g.getvertexfromedge(edges[0])
        C += [u.key, v.key]
        i = 0
        while i < len(edges):
            edge = edges[i]
            if edge.vertex1.key == u.key or edge.vertex1.key == v.key or edge.vertex2.key == u.key or edge.vertex2.key == v.key:
                edges.pop(i)
            else:
                i += 1
    return C

def test_approx_vertex_cover():
    """
    测试顶点覆盖问题的近似算法
    """
    g = Graph()
    g.addvertex(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    g.addedge('a', 'b')
    g.addedge('b', 'c')
    g.addedge('c', 'd')
    g.addedge('c', 'e')
    g.addedge('d', 'e')
    g.addedge('d', 'f')
    g.addedge('d', 'g')
    g.addedge('e', 'f')
    g.printadj()
    print(approx_vertex_cover(g))

def test():
    """
    测试函数
    """
    test_approx_vertex_cover()

if __name__ == '__main__':
    test()
else:
    pass
