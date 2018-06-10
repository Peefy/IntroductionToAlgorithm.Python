
class BTreeNode:
    '''
    B树结点
    '''
    def __init__(self, n = 0, key = None, keys = [], children = [], leaf = False, p = None):
        '''
        B树结点
        '''
        self.n = n
        self.key = None
        self.keys = keys
        self.children = children
        self.leaf = leaf
        self.p = p

    def __str__(self):
        return 'key:' + str(self.key) + ','\
            'children_n:' + str(self.n)

    def diskread(self):
        '''
        磁盘读
        '''
        pass

    def diskwrite(self):
        '''
        磁盘写
        '''
        pass

    @staticmethod
    def allocate_node():
        '''
        在O(1)时间内为一个新结点分配一个磁盘页

        假定由ALLOCATE-NODE所创建的结点无需做DISK-READ，因为磁盘上还没有关于该结点的有用信息

        Return
        ===
        `btreenode` : 分配的B树结点

        Example
        ===
        ```python
        btreenode = BTreeNode.allocate_node()
        ```
        '''
        return BTreeNode()

class BTree:
    '''
    B树
    '''
    def __init__(self):
        '''
        B树的定义
        '''
        self.lastnode: BTreeNode = None
        self.root: BTreeNode = None
        self.nodes = []

    @staticmethod
    def create():
        '''
        创建一棵空的B树

        Example
        ===
        ```python

        btree = BTree.create()

        ```
        '''
        node = BTreeNode.allocate_node()
        node.leaf = True
        node.n = 0
        node.key = 1
        btree = BTree()
        btree.root = node
        return btree

    def search(self, x : BTreeNode, k):
        '''
        搜索B树 : 寻找关键字为k的结点

        总的CPU时间为O(th)=O(tlogt(n)),h为树的高度，n为B树中所含的关键字个数，x.n < 2t

        Args
        ===
        `x` : BTreeNode 从结点x的子树开始搜索

        `k` : 搜索的关键字key

        Return
        ===
        `(y, i)` : 从结点`y`的第`i`个子女中找到了关键字为`k`的结点

        '''
        i = 0
        # 线性搜索过程，找出使k<=x.keys[i]成立的最小下标
        while i < x.n and k > x.keys[i]:
            i += 1
        # 如果找到了就返回有序对
        if i <= x.n and k == x.keys[i]:
            return (x, i)
        # 失败结束查找
        ## 如果x是一个叶结点
        if x.leaf == True:
            return None
        else:
            # x的子女执行磁盘读
            x.children[i].diskread()
            # 递归搜索x的适当子树
            return self.search(x.children[i], k)

