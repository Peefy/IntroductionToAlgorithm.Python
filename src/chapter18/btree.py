
class BTreeNode:
    '''
    B树结点
    '''
    def __init__(self, n = 0, isleaf = True):
        '''
        B树结点

        Args
        ===
        `n` : 结点包含关键字的数量

        `isleaf` : 是否是叶子节点

        '''
        # 结点包含关键字的数量
        self.n = n
        # 关键字的值数组
        self.keys = []
        # 子结点数组
        self.children = []
        # 是否是叶子节点
        self.isleaf = isleaf

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

    @classmethod
    def allocate_node(self):
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
    def __init__(self, m = 3):
        '''
        B树的定义
        '''
        self.M = m
        self.KEY_MAX = 2 * self.M - 1
        self.KEY_MIN = self.M - 1
        self.CHILD_MAX = self.KEY_MAX + 1
        self.CHILD_MIN = self.KEY_MIN + 1
        self.lastnode: BTreeNode = None
        self.root: BTreeNode = None
        self.nodes = []
        self.t = 4

    @staticmethod
    def create(m = 3):
        '''
        创建一棵空的B树

        Example
        ===
        ```python

        btree = BTree.create()

        ```
        '''
        node = BTreeNode.allocate_node()
        node.isleaf = True
        node.n = 0
        node.key = 1
        btree = BTree(m)
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

    def split_child(self, x : BTreeNode, i, y : BTreeNode):
        '''
        B树分裂

        满结点`y`按其中间关键字S的索引`i``进行分裂，S则提升到`y`的双亲结点`x`当中

        Args
        ===
        `x` : `y`的双亲结点

        `y` : 要分裂的结点

        `i` : 分裂结点`y`的分裂索引`i`
        '''
        t = self.t
        z = BTreeNode.allocate_node()
        z.leaf = y.leaf
        z.n = t - 1
        for j in range(t - 1):
            z.keys[j] = z.keys[j + t]
        if not y.leaf:
            for j in range(t):
                z.children[j] = z.children[j + t]
        y.n = t - 1
        for j in range(i + 1, x.n + 1):
            k = x.n + 1 - j + i + 1
            x.children[k + 1] = x.children[k]
        x.children[i + 1] = z
        for j in range(i, x.n):
            k = x.n - j + i
            x.keys[k + 1] = x.keys[k]
        x.keys[i] = y.keys[t]
        x.n = x.n + 1
        y.diskwrite()
        z.diskwrite()
        x.diskwrite()

    def insert(self, key):
        '''
        向B树中插入新结点`key`  
        '''
        pass

    def remove(self, key): 
        '''
        从B中删除结点`key`
        '''      
        pass
    
    def display(self):
        '''
        打印树的关键字  
        '''
        pass

    def contain(self, key):
        '''
        检查该`key`是否存在于B树中  
        '''
        pass

    def clear(self):
        '''
        清空B树  
        '''
        pass

    def __recursive_clear(self, pNode : BTreeNode):
        '''
        删除树  
        '''
        pass

    def __delete_node(self, pNode : BTreeNode):
        '''
        删除节点 
        '''
        pass
    
    def __search(self, pNode : BTreeNode, key):
        '''
        查找关键字  
        '''
        pass

    def __split_child(self, pParent : BTreeNode, nChildIndex, pChild : BTreeNode):
        '''
        分裂子节点
        '''
        pass
    
    def __insert_non_full(self, pNode: BTreeNode, key):
        '''
        在非满节点中插入关键字
        '''
        pass

    def __display_in_concavo(self, pNode: BTreeNode, count):
        '''
        用括号打印树 
        '''
        pass

    def __merge_child(self, pNode: BTreeNode, index):
        '''
        合并两个子结点
        '''
        pass

    def __recursive_remove(self, pNode: BTreeNode, key):
        '''
        递归的删除关键字`key`  
        '''
        raise NotImplementedError

    def predecessor(self, pNode: BTreeNode):
        '''
        前驱关键字
        '''
        pass

    def successor(self, pNode: BTreeNode):
        '''
        后继关键字
        '''
        raise NotImplementedError

def test():
    '''
    test class `BTree` and class `BTreeNode`
    '''
    tree = BTree.create()
    print(tree.root)

if __name__ == '__main__':
    test()
else:
    pass
