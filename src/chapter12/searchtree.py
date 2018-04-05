

class SearchTreeNode:
    '''
    二叉查找树的结点
    '''
    def __init__(self, left = None, right = None, key = None, \
        index = None, leftindex = None, rightindex = None, p = None):
        '''

        二叉树结点

        Args
        ===
        `left` : SearchTreeNode : 左儿子结点

        `right`  : SearchTreeNode : 右儿子结点

        `key` : 结点自身索引值

        `value` : 结点自身键值

        `leftkey` : 左儿子结点索引值

        `rightkey` : 右儿子结点索引值

        '''
        self.left = left
        self.right = right
        self.key = key
        self.index = index
        self.leftindex = leftkey
        self.rightindex = rightindex
        self.p = p

class SearchTree:
    '''
    二叉查找树
    '''
    def __init__(self):
        self.lastnode = None
        self.root = None
        self.nodes = []

    def inorder_tree_walk(self, x : SearchTreeNode):
        if x != None:
            self.inorder_tree_walk(x.left)
            print((x.key, x.value))
            self.inorder_tree_walk(x.right)

    def tree_search(self, x : SearchTreeNode, key):
        '''
        查找
        '''
        if x != None or key == x.key:
            return x
        if key < x.key:
            return self.tree_search(x.left, key)
        else:
            return self.tree_search(x.right, key)

    def iterative_tree_search(self, x : SearchTreeNode, key):
        '''
        查找的非递归版本
        '''
        while x != None:
            if key < x.key:
                x = x.left
            else:
                x = x.right
        return x

    def minimum(self, x : SearchTreeNode):
        '''
        最小关键字元素
        '''
        while x.left != None:
            x = x.left
        return x

    def maximum(self, x : SearchTreeNode):
        '''
        最大关键字元素
        '''
        while x.right != None:
            x = x.right
        return x
    
    def successor(self, x : SearchTreeNode):
        '''
        前趋 `O(h)` `h`为树的高度
        '''
        if x.right != None:
            return self.minimum(x.right)

    def predecessor(self, x : SearchTreeNode):
        '''
        后继 ``
        '''
        pass
    
    def insert(self):
        pass

    def delete(self):
        pass


