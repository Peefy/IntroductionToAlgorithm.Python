
import graph as _g
import notintersectset as _s
import math as _math
from copy import deepcopy as _deepcopy

class _MST:
    def __init__(self, *args, **kwwords):
        pass

    def generic_mst(self, g: _g.Graph):
        '''
        通用最小生成树算法

        Args
        ===
        `g` : 图G=(V,E)

        `w` : 图的权重

        Doc
        ===
        # A = []

        # while A does not form a spanning tree

        #    do find an edge (u,v) that is safe for A (保证不形成回路)

        #       A <- A ∪ {(u, v)}

        # return A

        '''
        A = ['generic mst']
        g.edges.sort()
        return A

    def mst_kruskal(self, g : _g.Graph):
        '''
        最小生成树的Kruska算法 时间复杂度`O(ElgV)`
        Args
        ===
        `g` : 图`G=(V,E)`

        Return
        ===
        `(mst_list, weight)` : 最小生成树列表和最小权重组成的`tuple`

        Example
        ===
        ```python
        g = Graph()
        g.clear()
        g.addvertex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
        g.addedgewithweight('a', 'b', 4)
        g.addedgewithweight('b', 'c', 8)
        g.addedgewithweight('c', 'd', 7)
        g.addedgewithweight('d', 'e', 9)
        g.addedgewithweight('a', 'h', 8)
        g.addedgewithweight('b', 'h', 11)
        g.addedgewithweight('c', 'i', 2)
        g.addedgewithweight('i', 'h', 7)
        g.addedgewithweight('h', 'g', 1)
        g.addedgewithweight('g', 'f', 2)
        g.addedgewithweight('f', 'e', 10)
        g.addedgewithweight('d', 'f', 14)
        g.addedgewithweight('c', 'f', 4)
        g.addedgewithweight('i', 'g', 4)
        mst_kruskal(g)
        >>> ([('h', 'g', 1), ('c', 'i', 2), ('g', 'f', 2), ('a', 'b', 4), ('c', 'f', 4), ('c', 'd', 7), ('b', 'c', 8), ('d', 'e', 9)], 37)
        ```
        '''
        s = _s.Set()
        A = []
        weight = 0
        for v in g.veterxs:
            s.make_set(v)
        g.edges.sort()
        for e in g.edges:
            (u, v) = g.getvertexfromedge(e)
            uset = s.find(u)
            vset = s.find(v)
            if uset != vset:
                A += [(u.key, v.key, e.weight)]
                s.union(uset, vset)
                weight += e.weight
        return A, weight

    def __change_weightkey_in_queue(self, Q, v, u):
        for q in Q:
            if q.key == v.key:
                q.weightkey = v.weightkey
                q.pi = u
                break

    def mst_prism(self, g : _g.Graph, r : _g.Vertex):
        '''
        最小生成树的Prism算法 时间复杂度`O(ElgV)`
        Args
        ===
        `g` : 图`G=(V,E)`

        Return
        ===
        `weight` : 最小权重
        '''
        for u in g.veterxs:
            u.isvisit = False
            u.weightkey = _math.inf
            u.pi = None
        if type(r) is not _g.Vertex:
            r = g.veterxs_atkey(r)
        else:
            r = g.veterxs_atkey(r.key)
        r.weightkey = 0   
        total_adj = g.getadj_from_matrix()
        weight = 0
        n = g.vertex_num
        weight_min = 0
        k = 0
        tree = []
        for j in range(n):
            weight_min = _math.inf
            u = None
            # 优先队列Q extract-min
            for v in g.veterxs:
                if v.isvisit == False and v.weightkey < weight_min:
                    weight_min = v.weightkey
                    u = v
            u.isvisit = True
            # 获取u的邻接表
            adj = g.getvertexadj(u)
            # 计算最小权重
            weight += weight_min        
            for v in adj:
                # 获取边
                edge = g.getedge(u, v)
                # 构造最小生成树
                if weight_min != 0 and edge.weight == weight_min:
                    tree.append((v.key, u.key, weight_min))
                # if v ∈ Q and w(u, v) < key[v]
                if v.isvisit == False and edge.weight < v.weightkey:
                    v.pi = u
                    v.weightkey = edge.weight
                    # 更新Vertex域 如果是引用则不需要，此处adj不是引用
                    for q in g.veterxs:
                        if q.key == v.key:
                            q.weightkey = v.weightkey
                            q.pi = v.pi
                            break
        return tree, weight

    def mst_dijkstra(self, g: _g.Graph, r: _g.Vertex):
        '''
        最小生成树的Prism算法 时间复杂度`O(ElgV)`
        Args
        ===
        `g` : 图`G=(V,E)`

        Return
        ===
        `weight` : 最小权重
        '''
        for u in g.veterxs:
            u.isvisit = False
            u.weightkey = _math.inf
            u.pi = None
        if type(r) is not _g.Vertex:
            r = g.veterxs_atkey(r)
        else:
            r = g.veterxs_atkey(r.key)
        r.weightkey = 0
        total_adj = g.getadj_from_matrix()
        weight = 0
        n = g.vertex_num
        weight_min = 0
        k = 0
        tree = []
        for j in range(n):
            weight_min = _math.inf
            u = None
            # 优先队列Q extract-min
            for v in g.veterxs:
                if v.isvisit == False and v.weightkey < weight_min:
                    weight_min = v.weightkey
                    u = v
            u.isvisit = True
            # 获取u的邻接表
            adj = g.getvertexadj(u)
            # 计算最小权重
            weight += weight_min
            for v in adj:
                # 获取边
                edge = g.getedge(u, v)
                # 构造最小生成树
                if weight_min != 0 and edge.weight == weight_min:
                    tree.append((v.key, u.key, weight_min))
                # if v ∈ Q and w(u, v) < key[v]
                if v.isvisit == False and edge.weight < v.weightkey:
                    v.pi = u
                    v.weightkey = edge.weight
                    # 更新Vertex域 如果是引用则不需要，此处adj不是引用
                    for q in g.veterxs:
                        if q.key == v.key:
                            q.weightkey = v.weightkey
                            q.pi = v.pi
                            break
        return tree, weight

__mst_instance = _MST()
generic_mst = __mst_instance.generic_mst
mst_kruskal = __mst_instance.mst_kruskal
mst_prism = __mst_instance.mst_prism
mst_dijkstra = __mst_instance.mst_dijkstra

def buildgraph():
    '''
    构造图
    '''
    g =  _g.Graph()
    g.clear()
    g.addvertex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
    g.addedgewithweight('a', 'h', 8)
    g.addedgewithweight('a', 'b', 4)
    g.addedgewithweight('b', 'c', 8)
    g.addedgewithweight('c', 'd', 7)
    g.addedgewithweight('d', 'e', 9)
    g.addedgewithweight('b', 'h', 11)
    g.addedgewithweight('c', 'i', 2)
    g.addedgewithweight('i', 'h', 7)
    g.addedgewithweight('h', 'g', 1)
    g.addedgewithweight('g', 'f', 2)
    g.addedgewithweight('f', 'e', 10)
    g.addedgewithweight('d', 'f', 14)
    g.addedgewithweight('c', 'f', 4)
    g.addedgewithweight('i', 'g', 6)
    return g

def test_mst_generic():
    g = _g.Graph()
    g.clear()
    g.addvertex(['a', 'b', 'c', 'd'])
    g.addedgewithweight('a', 'b', 2)
    g.addedgewithweight('a', 'd', 3)
    g.addedgewithweight('b', 'c', 1)
    print('邻接表为')
    _g._print_inner_conllection(g.adj)
    print('邻接矩阵为')
    print(g.matrix)
    print('图G=(V,E)的集合为')
    _g._print_inner_conllection(g.edges)
    print(generic_mst(g))
    print('边按权重排序后图G=(V,E)的集合为')
    _g._print_inner_conllection(g.edges)
    del g

def test_mst_kruskal():
    g = buildgraph()
    print('边和顶点的数量分别为:', g.edge_num, g.vertex_num)
    print('邻接表为')
    g.printadj()
    print('邻接矩阵为')
    print(g.matrix)
    print('最小生成树为：')
    mst_list = mst_kruskal(g)
    print(mst_list)
    del g

def test_mst_prism():
    g = buildgraph()
    print('边和顶点的数量分别为:', g.edge_num, g.vertex_num)
    print('邻接表为')
    g.printadj()
    print('邻接矩阵为')
    print(g.matrix)
    print('最小生成树为：')
    mst_list = mst_prism(g, 'a')
    print(mst_list)
    del g

def test_mst_dijkstra():
    g = buildgraph()
    print('邻接表为')
    g.printadj()
    print('邻接矩阵为：')
    print(g.matrix)
    del g

def test():
    '''
    测试函数
    '''
    test_mst_generic()
    test_mst_kruskal()
    test_mst_prism()
    test_mst_dijkstra()

if __name__ == '__main__':
    print('test as follows')
    test()
else:
    pass

