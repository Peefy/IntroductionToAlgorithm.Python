
## 用于不相交集合的数据结构

不相交集合数据结构保持一组不相交的动态集合S={S1,S2,...,Sk}，每个集合通过一个代表来识别，代表即集合中的某个成员。集合中的每一个元素是由一个对象表示的，设x表示一个对象，希望支持以下操作

* MAKE-SET(x)：其唯一成员(因而其代表)就是x。因为各集合是不相交的，故要求x没有在其他集合出现过

* UNION(x, y)：将包含x和y的动态集合(比如说Sx和Sy)合并为一个新的集合(并集)

* FIND-SET(x)：返回一个指针，指向包含x的(唯一)集合的代表

不相交集合数据结构有多种应用,其中之一是用于确定一个无向图中连通子图的个数

```python


class UndirectedGraph:
    '''
    无向图 `G=(V, E)`
    '''
    def __init__(self, vertexs : list = [], edges : list = []):
        '''
        无向图 `G=(V, E)`

        Args
        ===
        `vertexs` : 顶点集合 `list` contains element which contains one element denote a point

        `edges` : 边集合 `list` contains element which contains two elements denote one edge of two points repectively

        Example
        ===
        ```python
        import graph
        >>> g = graph.UndirectedGraph(['a', 'b', 'c', 'd'], [('a', 'b')])
        ```

        '''
        self.vertexs = vertexs
        self.edges = edges
        self.__findcount = 0
        self.__unioncount = 0
        self.__kcount = 0

    def get_connected_components(self):
        '''
        获取无向图中连通子图的集合
        '''
        self.__findcount = 0
        self.__unioncount = 0
        self.__kcount = 0
        set = Set()
        for v in self.vertexs:
            set.make_set(v)
        for e in self.edges:
            u, v = e
            set1 = set.find(u)
            set2 = set.find(v)
            self.__findcount += 2
            if set1 != set2:
                set.union(set1, set2)
                self.__unioncount += 1
        self.__kcount = len(set.sets)
        return set

    def print_last_connected_count(self):
        '''
        获取上一次连接无向图之后调用函数情况
        '''
        print('the k num:{} the find num:{} the union num:{}'. \
            format(self.__kcount, self.__findcount, self.__unioncount))

class Set:
    '''
    不相交集合数据结构
    '''
    def __init__(self):
        '''
        不相交集合数据结构
        '''
        self.sets = []

    def make_set(self, element):
        '''
        建立一个新的集合
        '''
        self.sets.append({element})

    def union(self, set1, set2):
        '''
        将子集合`set1`和`set2`合并
        '''
        if set1 is None or set2 is None:
            return
        self.sets.remove(set1)
        self.sets.remove(set2)
        self.sets.append(set1 | set2)

    def find(self, element):
        '''
        找出包含元素`element`的集合
        '''
        for set in self.sets:
            if element in set:
                return set
        return None
    
    def __str__(self):
        return str(self.sets)


def connected_components(g: UndirectedGraph):
    '''
    求一个无向图中连通子图的个数

    Args
    ===
    `g` : UndirectedGraph 无向图

    '''
    set = Set()
    for v in g.vertexs:
        set.make_set(v)
    for e in g.edges:
        u, v = e
        set1 = set.find(u)
        set2 = set.find(v)
        if set1 != set2:
            set.union(set1, set2)
    return set

def test_graph_connected():
    '''
    test_graph_connected
    '''
    g = UndirectedGraph()
    g.vertexs = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    g.edges.append(('b', 'd'))
    g.edges.append(('e', 'g'))
    g.edges.append(('a', 'c'))
    g.edges.append(('h', 'i'))
    g.edges.append(('a', 'b'))
    g.edges.append(('e', 'f'))
    g.edges.append(('b', 'c'))
    print(g.get_connected_components())
    g.print_last_connected_count()

if __name__ == '__main__':
    test_graph_connected()
else:
    pass    


```

[Github Code](https://github.com/Peefy/CLRS_dugu_code-master/blob/master/src/chapter21)
