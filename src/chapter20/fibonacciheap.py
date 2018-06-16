
class FibonacciHeapNode:
    '''
    斐波那契堆结点
    '''
    def __init__(self, key = None, degree = None, p = None, child = None, \
        left = None, right = None, mark = None):
        '''
        斐波那契堆结点

        Args
        ===
        `key` : 关键字值

        `degree` : 子结点的个数

        `p` : 父结点

        `child` : 任意一个子结点

        `left` : 左兄弟结点

        `right` : 右兄弟结点

        `mark` : 自从x上一次成为另一个结点子女以来，它是否失掉了一个孩子

        '''
        self.key = key
        self.degree = degree
        self.p = p
        self.child = child
        self.left = left
        self.right = right
        self.mark = mark

class FibonacciHeap:
    '''
    斐波那契堆 不涉及删除元素的可合并堆操作仅需要`O(1)`的平摊时间,这是斐波那契堆的优点
    '''
    def __init__(self):
        '''
        斐波那契堆 不涉及删除元素的可合并堆操作仅需要`O(1)`的平摊时间,这是斐波那契堆的优点
        '''
        self.min = None
        self.n = 0
        self.phi = 0
        self.t = 0
        self.m = 0

    @classmethod
    def make_fib_heap(self):
        '''
        创建一个新的空的斐波那契堆

        Example
        ===
        ```python
        heap = FibonacciHeap.make_fib_heap()
        ```
        '''
        heap = FibonacciHeap()
        return heap
    
    def insertkey(self, key):
        '''
        插入一个关键字为`key`结点`node`
        '''
        node = FibonacciHeapNode(key)
        self.insert(node)

    def insert(self, node : FibonacciHeapNode):
        '''
        插入一个结点`node`
        '''
        pass
    
    