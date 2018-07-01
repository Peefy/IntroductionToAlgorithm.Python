
if __name__ == '__main__':
    import graph as _g
else:
    from . import graph as _g

def generic_mst(g : _g.Graph, w):
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
    return A

def test():
    '''
    测试函数
    '''
    w = 2
    g = _g.Graph()
    g.clear()
    g.addvertex(['a', 'b', 'c'])
    g.addedge('a', 'b')
    g.addedge('b', 'c')
    print('邻接表为')
    _g._print_inner_conllection(g.adj)
    print('邻接矩阵为')
    print(g.matrix)
    print(generic_mst(g, w))

if __name__ == '__main__':
    print('test as follows')
    test()
else:
    pass

