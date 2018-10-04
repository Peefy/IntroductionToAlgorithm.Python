
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

def trim(L, d):
    """
    对列表`L`的修整
    """  
    m = len(L)
    L_ = [L[0]]
    last = L[0]
    for i in range(1, m):
        if L[i] > last * (1 + d):
            L_.append(L[i])
            last = L[i]
    return L_

def approx_subset_sum(S, t, e):
    n = len(S)
    l_last = [0]
    for i in range(n):
        l = sorted(l_last + list(map(lambda x : x + S[i], l_last)))
        l = trim(l, e / (2 * n))
        l_last.clear()
        for num in l:
            if num <= t:
                l_last.append(num)
    return max(l_last)

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
    print(trim([10, 11, 12, 15, 20, 21, 22, 23, 24, 29], 0.1))
    print(approx_subset_sum([104, 102, 201, 101], 308, 0.4))

if __name__ == '__main__':
    test()
else:
    pass
