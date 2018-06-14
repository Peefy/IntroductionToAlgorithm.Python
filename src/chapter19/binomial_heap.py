
class BinomialHeapNode:
    '''
    二项堆结点
    '''
    def __init__(self, p = None, key = None, degree = None, 
        child = None, sibling = None):
        '''
        二项堆结点

        Args
        ===
        `p` : 父结点
        
        `key` : 关键字

        `degree` : 子结点的个数

        `child` : 子结点

        `sibling` : 二项堆同根的下一个兄弟

        '''
        self.p = p
        self.key = key
        self.degree = degree
        self.child = child
        self.sibling = sibling
    
    def __str__(self):
        return str(self.key)

class BinomialHeap:
    '''
    二项堆
    '''
    def __init__(self, head : BinomialHeapNode = None):
        '''
        二项堆

        Args
        ===
        `head` : 头结点

        '''
        self.head = head

    def minimum(self):
        '''
        求出指向包含n个结点的二项堆H中具有最小关键字的结点

        时间复杂度`O(lgn)`

        '''
        y = None
        x = self.head
        min = -2147483648
        while x != None:
            if x.key < min:
                min = x.key
                y = x
            x = x.sibling
        return y

    def link(self, y : BinomialHeapNode, z : BinomialHeapNode):
        '''
        将一结点`y`为根的Bk-1树与以结点`z`为根的Bk-1树连接起来
        也就是使`z`成为`y`的父结点，并且成为一棵Bk树

        时间复杂度`O(1)`

        Args
        ===
        `y` : 一个结点

        `z` : 另外一个结点
        '''
        y.p = z
        y.sibling = z.child
        z.child = y
        z.degree += 1

    def extract_min(self):
        '''
        '''
        pass

    @classmethod
    def make_heap(self):
        heap = BinomialHeap()
        return heap

    @classmethod
    def merge(self):
        pass

    @classmethod
    def union(self):
        pass

if __name__ == '__main__':
    heap = BinomialHeap.make_heap()
    if heap.head is not None:
        print(heap.head)
else:
    pass
