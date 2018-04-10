
RED = 1
BLACK = 0

class RedBlackTreeNode:
    def __init__(self, key, index, color, \
        p = None, left = None, right = None):
        '''
        红黑树树结点

        Args
        ===
        `left` : SearchTreeNode : 左儿子结点

        `right`  : SearchTreeNode : 右儿子结点

        `index` : 结点自身索引值

        `key` : 结点自身键值

        `p` : 父节点

        '''
        self.left = left
        self.right = right
        self.key = key
        self.index = index
        self.color = color
        self.p = p

    def __str__(self):
        return 'key:' + str(self.key) + ','\
                'index:' + str(self.index)

class RedBlackTree:
    '''
    红黑树
    '''

    def __init__(self):
        self.lastnode : RedBlackTreeNode = None
        self.root : RedBlackTreeNode = None
        self.nodes = []

    def leftrotate(self, x : RedBlackTreeNode):
        '''
        左旋
        '''
        y : RedBlackTreeNode = x.right
        x.right = y.left
        y.left.p = x
        y.p = x.p
        if x.p == None:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.left = x
        x.p = y
            

