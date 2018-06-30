
if __name__ == '__main__':
    import graph as _g
else:
    from . import graph as _g

def generic_mst():
    '''
    通用最小生成树算法
    '''
    pass

def test():
    '''
    测试函数
    '''
    g = _g.Graph()
    g.clear()
    g.addvertex('a')
    g.addvertex('b')
    g.addvertex('c')
    g.addedge('a', 'b')
    g.addedge('b', 'c')
    print(g.adj)

if __name__ == '__main__':
    print('test as follows')
    test()
else:
    pass

