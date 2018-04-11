
RED = 1
BLACK = 0

class RedBlackTreeNode:
    '''
    红黑树结点
    '''
    def __init__(self, key, index, color = RED, \
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
        '''
        红黑树
        '''
        self.lastnode : RedBlackTreeNode = None
        self.root : RedBlackTreeNode = None
        self.nodes = []
        self.nil = None

    def insertkeyandcolor(self, key, index, color = RED):
        z = RedBlackTreeNode(key, index, color)
        self.insert(z)

    def insert(self, z : RedBlackTreeNode):
        '''
        插入红黑树结点 时间复杂度`O(lgn)`
        '''
        y = self.nil
        x = self.root
        while x != self.nil:
            y = x
            if z.key < x.key:
                x = x.left
            else:
                x = x.right
        z.p = y
        if y == self.nil:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z
        z.left = self.nil
        z.right = self.nil
        z.color = RED
        self.insert_fixup(z)

    def insert_fixup(self, z : RedBlackTreeNode):
        '''
        修正红黑树性质，结点重新旋转和着色
        '''
        while z.p.color == RED:
            if z.p == z.p.p.left:
                y = z.p.p.right
                if y.color == RED:
                    z.p.color = BLACK
                    y.color = BLACK
                    z.p.p.color = RED
                    z = z.p.p
                elif z == z.p.right:
                    z = z.p
                    self.leftrotate(z)
                z.p.color = BLACK
                z.p.p.color = RED
                self.rightrotate(z)
            else:
                y = z.p.p.left
                if y.color == RED:
                    z.p.color = BLACK
                    y.color = BLACK
                    z.p.p.color = RED
                    z = z.p.p
                elif z == z.p.right:
                    z = z.p
                    self.leftrotate(z)
                z.p.color = BLACK
                z.p.p.color = RED
                self.rightrotate(z)               
        self.root.color = BLACK    
        
    def leftrotate(self, x : SearchTreeNode):
        '''
        左旋 时间复杂度:`O(1)`
        '''
        if x.right == None:
            return
        y : SearchTreeNode = x.right
        x.right = y.left
        if y.left != None:
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

    def rightrotate(self, x : SearchTreeNode):
        '''
        右旋 时间复杂度:`O(1)`
        '''
        if x.left == None:
            return
        y : SearchTreeNode = x.left
        x.left = y.right
        if y.right != None:
            y.right.p = x
        y.p = x.p
        if x.p == None:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.right = x
        x.p = y
            

