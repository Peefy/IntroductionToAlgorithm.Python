---
layout: post
title: "用Python实现B树"
description: "用Python实现的B树"
categories: [Python]
tags: [python]
redirect_from:
  - /2018/06/10/
---

## B树

B树是为磁盘或其他直接存取辅助设备而设计的一种平衡查找树

B树与红黑树的主要不同在于，B树的结点可以有许多子女，从几个到几千个，就是说B树的分支因子可能很大

```python

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

        returnStr = 'keys:['
        for i in range(self.n):
            returnStr += str(self.keys[i]) + ' '
        returnStr += '];childrens:['
        for child in self.children:
            returnStr += str(child) + ';'
        returnStr += ']\r\n'
        return returnStr

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
    def allocate_node(self, key_max):
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
        node = BTreeNode()
        child_max = key_max + 1
        for i in range(key_max):
            node.keys.append(None)
        for i in range(child_max):
            node.children.append(None)
        return node

class BTree:
    '''
    B树
    '''
    def __init__(self, m = 3):
        '''
        B树的定义
        '''
        # B树的最小度数
        self.M = m
        # 节点包含关键字的最大个数
        self.KEY_MAX = 2 * self.M - 1
        # 非根结点包含关键字的最小个数
        self.KEY_MIN = self.M - 1
        # 子结点的最大个数
        self.CHILD_MAX = self.KEY_MAX + 1
        # 子结点的最小个数
        self.CHILD_MIN = self.KEY_MIN + 1
        # 根结点
        self.root: BTreeNode = None

    def __new_node(self):
        '''
        创建新的B树结点
        '''
        return BTreeNode.allocate_node(self.KEY_MAX)

    def insert(self, key):
        '''
        向B树中插入新结点`key`  
        '''
        # 检查关键字是否存在
        if self.contain(key) == True:
            return False
        else:
            # 检查是否为空树
            if self.root is None:
                node = self.__new_node()
                node.diskwrite()
                self.root = node    
            # 检查根结点是否已满      
            if self.root.n == self.KEY_MAX:
                # 创建新的根结点
                pNode = self.__new_node()
                pNode.isleaf = False
                pNode.children[0] = self.root
                self.__split_child(pNode, 0, self.root)
                # 更新结点指针
                self.root = pNode
            self.__insert_non_full(self.root, key)
            return True

    def remove(self, key): 
        '''
        从B中删除结点`key`
        '''      
        # 如果关键字不存在
        if not self.search(self.root, key):
            return False
        # 特殊情况处理
        if self.root.n == 1:
            if self.root.isleaf == True:
                self.clear()
            else:
                pChild1 = self.root.children[0]
                pChild2 = self.root.children[1]
                if pChild1.n == self.KEY_MIN and pChild2.n == self.KEY_MIN:
                    self.__merge_child(self.root, 0)
                    self.__delete_node(self.root)
                    self.root = pChild1
        self.__recursive_remove(self.root, key)
        return True
    
    def display(self):
        '''
        打印树的关键字  
        '''
        self.__display_in_concavo(self.root, self.KEY_MAX * 10)

    def contain(self, key):
        '''
        检查该`key`是否存在于B树中  
        '''
        self.__search(self.root, key)

    def clear(self):
        '''
        清空B树  
        '''
        self.__recursive_clear(self.root)
        self.root = None

    def __recursive_clear(self, pNode : BTreeNode):
        '''
        删除树  
        '''
        if pNode is not None:
            if not pNode.isleaf:
                for i in range(pNode.n):
                    self.__recursive_clear(pNode.children[i])
            self.__delete_node(pNode)

    def __delete_node(self, pNode : BTreeNode):
        '''
        删除节点 
        '''
        if pNode is not None:
            pNode = None
    
    def __search(self, pNode : BTreeNode, key):
        '''
        查找关键字  
        '''
        # 检测结点是否为空，或者该结点是否为叶子节点
        if pNode is None:
            return False
        else:
            i = 0
            # 找到使key < pNode.keys[i]成立的最小下标
            while i < pNode.n and key > pNode.keys[i]:
                i += 1
            if i < pNode.n and key == pNode.keys[i]:
                return True
            else:
                # 检查该结点是否为叶子节点
                if pNode.isleaf == True:
                    return False
                else:
                    return self.__search(pNode.children[i], key)

    def __split_child(self, pParent : BTreeNode, nChildIndex, pChild : BTreeNode):
        '''
        分裂子节点
        '''
        # 将pChild分裂成pLeftChild和pChild两个结点
        pRightNode = self.__new_node()  # 分裂后的右结点
        pRightNode.isleaf = pChild.isleaf
        pRightNode.n = self.KEY_MIN
        # 拷贝关键字的值
        for i in range(self.KEY_MIN):
            pRightNode.keys[i] = pChild.keys[i + self.CHILD_MIN]
        # 如果不是叶子结点，就拷贝孩子结点指针
        if not pChild.isleaf:
            for i in range(self.CHILD_MIN):
                pRightNode.children[i] = pChild.children[i + self.CHILD_MIN]
        # 更新左子树的关键字个数
        pChild.n = self.KEY_MIN
        # 将父结点中的pChildIndex后的所有关键字的值和子树指针向后移动一位
        for i in range(nChildIndex, pParent.n):
            j = pParent.n + nChildIndex - i
            pParent.children[j + 1] = pParent.children[j]
            pParent.keys[j] = pParent.keys[j - 1]
        # 更新父结点的关键字个数
        pParent.n += 1
        # 存储右子树指针
        pParent.children[nChildIndex + 1] = pRightNode
        # 把结点的中间值提到父结点
        pParent.keys[nChildIndex] = pChild.keys[self.KEY_MIN]
        pChild.diskwrite()
        pRightNode.diskwrite()
        pParent.diskwrite()
    
    def __insert_non_full(self, pNode: BTreeNode, key):
        '''
        在非满节点中插入关键字
        '''
        # 获取结点内关键字个数
        i = pNode.n
        # 如果pNode是叶子结点
        if pNode.isleaf == True:
            # 从后往前 查找关键字的插入位置
            while i > 0 and key < pNode.keys[i - 1]:
                # 向后移位
                pNode.keys[i] = pNode.keys[i - 1]
                i -= 1
            # 插入关键字的值
            pNode.keys[i] = key
            # 更新结点关键字的个数
            pNode.n += 1
            pNode.diskwrite()
        # pnode是内结点
        else:
            # 从后往前 查找关键字的插入的子树
            while i > 0 and key < pNode.keys[i - 1]:
                i -= 1
            # 目标子树结点指针
            pChild = pNode.children[i]
            pNode.children[i].diskread()
            # 子树结点已经满了
            if pChild.n == self.KEY_MAX:
                # 分裂子树结点
                self.__split_child(pNode, i, pChild)
                # 确定目标子树
                if key > pNode.keys[i]:
                    pChild = pNode.children[i + 1]
            # 插入关键字到目标子树结点
            self.__insert_non_full(pChild, key)

    def __display_in_concavo(self, pNode: BTreeNode, count):
        '''
        用括号打印树 
        '''
        if pNode is not None:
            i = 0
            j = 0
            for i in range(pNode.n):
                if not pNode.isleaf:
                    self.__display_in_concavo(pNode.children[i], count - 2)
                for j in range(-1, count):
                    k = count - j - 1
                    print('-', end='')
                print(pNode.keys[i])
            if not pNode.isleaf:
                self.__display_in_concavo(pNode.children[i], count - 2)

    def __merge_child(self, pParent: BTreeNode, index):
        '''
        合并两个子结点
        '''
        pChild1 = pParent.children[index]
        pChild2 = pParent.children[index + 1]
        # 将pChild2数据合并到pChild1
        pChild1.n = self.KEY_MAX
        # 将父结点index的值下移
        pChild1.keys[self.KEY_MIN] = pParent.keys[index]
        for i in range(self.KEY_MIN):
            pChild1.keys[i + self.KEY_MIN + 1] = pChild2.keys[i]
        if not pChild1.isleaf:
            for i in range(self.CHILD_MIN):
                pChild1.children[i + self.CHILD_MIN] = pChild2.children[i]
        # 父结点删除index的key，index后的往前移一位
        pParent.n -= 1
        for i in range(index, pParent.n):
            pParent.keys[i] = pParent.keys[i + 1]
            pParent.children[i + 1] = pParent.children[i + 2]
        # 删除pChild2
        self.__delete_node(pChild2)

    def __recursive_remove(self, pNode: BTreeNode, key):
        '''
        递归的删除关键字`key`  
        '''
        i = 0
        while i < pNode.n and key > pNode.keys[i]:
            i += 1
        # 关键字key在结点pNode
        if i < pNode.n and key == pNode.keys[i]:
            # pNode是个叶结点
            if pNode.isleaf == True:
                # 从pNode中删除k
                for j in range(i, pNode.n):
                    pNode.keys[j] = pNode.keys[j + 1]
                return
            # pNode是个内结点
            else:
                # 结点pNode中前于key的子结点
                pChildPrev = pNode.children[i]
                # 结点pNode中后于key的子结点
                pChildNext = pNode.children[i + 1]
                if pChildPrev.n >= self.CHILD_MIN:
                    # 获取key的前驱关键字
                    prevKey = self.predecessor(pChildPrev)
                    self.__recursive_remove(pChildPrev, prevKey)
                    # 替换成key的前驱关键字
                    pNode.keys[i] = prevKey
                    return
                # 结点pChildNext中至少包含CHILD_MIN个关键字
                elif pChildNext.n >= self.CHILD_MIN:
                    # 获取key的后继关键字
                    nextKey = self.successor(pChildNext)
                    self.__recursive_remove(pChildNext, nextKey)
                    # 替换成key的后继关键字
                    pNode.keys[i] = nextKey
                    return
                # 结点pChildPrev和pChildNext中都只包含CHILD_MIN-1个关键字
                else:
                    self.__merge_child(pNode, i)
                    self.__recursive_remove(pChildPrev, key)
        # 关键字key不在结点pNode中
        else:
            # 包含key的子树根结点
            pChildNode = pNode.children[i]
            # 只有t-1个关键字
            if pChildNode.n == self.KEY_MAX:
                # 左兄弟结点
                pLeft = None
                # 右兄弟结点
                pRight = None
                # 左兄弟结点
                if i > 0:
                    pLeft = pNode.children[i - 1]
                # 右兄弟结点
                if i < pNode.n:
                    pRight = pNode.children[i + 1]
                j = 0
                if pLeft is not None and pLeft.n >= self.CHILD_MIN:
                    # 父结点中i-1的关键字下移至pChildNode中
                    for j in range(pChildNode.n):
                        k = pChildNode.n - j
                        pChildNode.keys[k] = pChildNode.keys[k - 1]
                    pChildNode.keys[0] = pNode.keys[i - 1]
                    if not pLeft.isleaf:
                        # pLeft结点中合适的子女指针移到pChildNode中
                        for j in range(pChildNode.n + 1):
                            k = pChildNode.n + 1 - j
                            pChildNode.children[k] = pChildNode.children[k - 1]
                        pChildNode.children[0] = pLeft.children[pLeft.n]
                    pChildNode.n += 1
                    pNode.keys[i] = pLeft.keys[pLeft.n - 1]
                    pLeft.n -= 1
                # 右兄弟结点至少有CHILD_MIN个关键字
                elif pRight is not None and pRight.n >= self.CHILD_MIN:
                    # 父结点中i的关键字下移至pChildNode中
                    pChildNode.keys[pChildNode.n] = pNode.keys[i]
                    pChildNode.n += 1
                    # pRight结点中的最小关键字上升到pNode中
                    pNode.keys[i] = pRight.keys[0]
                    pRight.n -= 1
                    for j in range(pRight.n):
                        pRight.keys[j] = pRight.keys[j + 1]
                    if not pRight.isleaf:
                        # pRight结点中合适的子女指针移动到pChildNode中
                        pChildNode.children[pChildNode.n] = pRight.children[0]
                        for j in range(pRight.n):
                            pRight.children[j] = pRight.children[j + 1]
                # 左右兄弟结点都只包含CHILD_MIN-1个结点
                elif pLeft is not None:
                    self.__merge_child(pNode, i - 1)
                    pChildNode = pLeft
                # 与右兄弟合并
                elif pRight is not None:
                    self.__merge_child(pNode, i)
            self.__recursive_remove(pChildNode, key)

    def predecessor(self, pNode: BTreeNode):
        '''
        前驱关键字
        '''
        while not pNode.isleaf:
            pNode = pNode.children[pNode.n]
        return pNode.keys[pNode.n - 1]

    def successor(self, pNode: BTreeNode):
        '''
        后继关键字
        '''
        while not pNode.isleaf:
            pNode = pNode.children[0]
        return pNode.keys[0]

def test():
    '''
    test class `BTree` and class `BTreeNode`
    '''
    tree = BTree(3)
    tree.insert(11)
    tree.insert(3)
    tree.insert(1)
    tree.insert(4)
    tree.insert(33)
    tree.insert(13)
    tree.insert(63)
    tree.insert(43)
    tree.insert(2)
    print(tree.root)
    tree.display()
    tree.clear()
    tree = BTree(2)
    tree.insert(11)
    tree.insert(3)
    tree.insert(1)
    tree.insert(4)
    tree.insert(33)
    tree.insert(13)
    tree.insert(63)
    tree.insert(43)
    tree.insert(2)
    print(tree.root)
    tree.display()

if __name__ == '__main__':
    test()
else:
    pass

```

[Github Code](https://github.com/Peefy/CLRS_dugu_code-master/blob/master/src/chapter18)
