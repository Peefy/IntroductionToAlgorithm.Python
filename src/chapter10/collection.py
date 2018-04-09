import json as _json
from random import randint as _randint
class Stack:
    '''
    栈
    '''
    def __init__(self, iterable = None):
        self.__top = -1
        self.array = []
        if iterable != None:
            self.array = list(iterable)
    
    def isEmpty(self): 
        '''
        栈是否为空

        Return
        ===
        `isempty` -> bool
        '''
        return self.__top == -1

    def push(self, item):
        '''
        入栈操作
        '''
        self.__top = self.__top + 1
        self.array.append(item)

    def pop(self):
        '''
        出栈操作
        '''
        if self.isEmpty() == True:
            raise Exception('the stack has been empty')
        else:
            self.__top = self.__top - 1
            return self.array.pop()

    def count(self):
        '''
        返回栈中所有元素的总数
        '''
        return len(self.array)

class TwoStack:
    '''
    用一个数组实现的两个栈
    '''
    def __init__(self, size = 5):
        self.__one_top = -1
        self.__two_top = size
        self.__size = size
        self.__array = list(range(size))    
    
    def one_push(self, item):
        self.__judgeisfull()
        self.__one_top += 1
        self.__array[self.__one_top] = item

    def one_pop(self):
        self.__judgeisempty()
        x = self.__array[self.__one_top]
        self.__one_top -= 1
        return x

    def two_push(self, item):
        self.__judgeisfull()
        self.__two_top -= 1
        self.__array[self.__two_top] = item

    def two_pop(self):
        self.__judgeisempty()
        x = self.__array(self.__two_top)
        self.__two_top += 1
        return x

    def one_all(self):
        array = []
        if self.__one_top != -1:
            for i in range(self.__one_top):
                array.append(self.__array[i])
        return array

    def two_all(self):
        array = []
        if self.__two_top != self.__size:
            for i in range(self.__two_top, self.__size):
                index = self.__size + self.__two_top - i - 1
                array.append(self.__array[index])
        return array

    def __judgeisfull(self):
        if self.__one_top + 1 == self.__two_top:
            raise Exception('Exception: stack is full!')

    def __judgeisempty(self):
        if self.__one_top == -1 or self.__two_top == self.__size:
            raise Exception('stack is full!')
class StackUsingQueue:
    '''
    用队列实现的栈
    '''
    def __init__(self, iterable = None):
        self.__queue1 = Queue()
        self.__queue2 = Queue()
    
    def push(self, item):
        self.__queue1.enqueue(item)

    def pop(self):
        for i in range(self.__queue1.length() - 1):
            self.__queue2.enqueue(self.__queue1.dequeue())
        x = self.__queue1.dequeue()
        for i in range(self.__queue2.length()):
            self.__queue1.enqueue(self.__queue2.dequeue())
        return x
        
    def count(self):
        return self.__queue1.length()
class Queue:
    '''
    队列
    '''
    def __init__(self, iterable = None):
        self.tail = 0
        self.array = []
        if iterable != None:
            self.array = list(iterable)

    def enqueue(self, item):
        '''
        元素`item`加入队列
        '''
        self.array.append(item)
        if self.tail == self.length:
            self.tail = 0
        else:
            self.tail = self.tail + 1

    def dequeue(self):
        '''
        元素出队列
        '''
        if self.length() == 0:
            raise Exception('Exception: the queue has been empty')
        x = self.array[0]
        self.array.remove(x)
        return x

    def length(self):
        return len(self.array)
class DoubleQueue:
    '''
    双向队列
    '''
    def __init__(self, iterable = None):
        self.tail = 0
        self.array = []
        if iterable != None:
            self.array = list(iterable)

    def enqueue(self, item):
        '''
        元素`item`加入队列
        '''
        self.array.append(item)
        if self.tail == self.length:
            self.tail = 0
        else:
            self.tail = self.tail + 1

    def dequeue(self):
        '''
        元素出队列
        '''
        if self.length() == 0:
            raise Exception('Exception: the queue has been empty')
        x = self.array[0]
        self.array.remove(x)
        return x

    def enqueue_reverse(self, item):
        self.array.insert(0, item)

    def dequeue_reverse(self):
        self.array.pop()

    def length(self):
        return len(self.array)

class QueueUsingStack:
    '''
    用栈实现的队列
    '''
    def __init__(self, iterable = None):
        self.__stack1 = Stack()
        self.__stack2 = Stack()
    
    def enqueue(self, item):
        self.__stack1.push(item)

    def dequeue(self):
        for i in range(self.__stack1.count() - 1):
            self.__stack2.push(self.__stack1.pop())
        x = self.__stack1.pop()
        for i in range(self.__stack2.count()):
            self.__stack1.push(self.__stack2.pop())
        return x
    
    def count(self):
        return self.__stack1.count()

class ListNode:
    '''
    链表节点
    '''
    def __init__(self, value = None):
        '''
        链表节点
        ```python
        >>> ListNode() 空节点   
        >>> ListNode(value) 值为value的链表节点
        ```
        '''
        self.value = value
        self.key = -1      
        self.prev = None
        self.next = None

    def __str__(self):
        return "key:" + str(self.key) + ";value:" + str(self.value)

    def getisNone(self):
        '''
        链表节点是否为空
        '''
        return self.key == None

    isNone = property(getisNone, None)

class List:   
    '''
    链表
    '''    
    def __init__(self):
        '''
        初始化一个空链表
        '''
        self.head = None
        self.tail = None
        self.next = None
        self.__length = 0

    def search(self, k):
        '''
        找出键值为k的链表节点元素，最坏情况为`Θ(n)`
        '''
        x = self.head
        while x.value != None and x.key != k:
            x = x.next
        return x

    def get_random_node(self) -> ListNode:
        num = _randint(0, self.count() - 1)
        j = self.head
        for iterate in range(num):
            j = j.next
        return j

    def compact_search(self, k):
        '''
        已经排序的链表中找出键值为k的链表节点元素，期望情况为`O(sqrt(n))`

        Args
        ===
        `k` : 待寻找元素的键值

        '''
        n = self.count()
        i = self.head
        while i != None and i.key > k:
            num = _randint(0, n - 1)
            j = self.get_random_node()
            if i.key < j.key and j.key <= k:
                i = j
                if i.key == k:
                    return i
            i = i.next
        if i == None or i.key < k:
            return None
        else:
            return i

    def compact_list_search(self, k, t):
        '''
        已经排序的链表中找出键值为k的链表节点元素，期望情况为`O(sqrt(n))`

        Args
        ===
        `k` : 待寻找元素的键值

        `t` : 循环迭代次数上界

        '''
        i = self.head
        for q in range(t):
            j = self.get_random_node()
            if i.key > j.key:
                i = j
                if i.key == k:
                    return i
        while i != None and i.key < k:
            i = i.next
        if i != None and i.key > k:
            return None
        else:
            return i

    def findtail(self):
        x = self.head
        while x != None and x.value != None:
            prev = x
            x = x.next
        return prev

    def insert(self, x):
        '''
        链表插入元素x
        '''
        self.__insert(ListNode(x))

    def __insert(self, x : ListNode):
        # 插入的元素按增量键值去
        x.key = self.__length;   
        # 把上一个头节点放到下一个节点去   
        x.next = self.head
        # 判断是否第一次插入元素
        if self.head != None and self.head.isNone == False:
            self.head.prev = x
        # 新插入的元素放到头节点去
        self.head = x
        # 新插入的节点前面没有元素
        x.prev = None
        self.__increse_length()

    def delete(self, item, key):
        '''
        链表删除元素x
        '''
        if type(item) is not ListNode:
            x = ListNode(item)
            x.key = key
        else:
            x = item
        if x.prev != None and x.prev.isNone == False:
            x.prev.next = x.next
        else:
            self.head = x.next
        if x.next != None and x.next.isNone == False:
            x.next.prev = x.prev
        self.__length -= 1

    def delete_bykey(self, k : int) -> ListNode:
        '''
        根据键值删除元素
        '''
        x = self.search(k)
        self.delete(x, x.key)
        return x.value

    def count(self):
        '''
        返回链表中元素的数量总和
        '''
        return self.__length

    def all(self):
        '''
        返回链表中所有元素组成的集合
        '''
        array = []
        x = self.head
        count = self.count()
        while x != None:
            value = x.value
            if value != None:
                array.append(value)
            x = x.next
        array.reverse()
        return array

    def __increse_length(self):
        self.__length += 1       

    def __reduce_length(self):
        self.__length -= 1

class QueueUsingList:
    '''
    使用链表构造的队列
    '''
    def __init__(self):
        self.__list = List()
        self.__length = 0

    def enqueue(self, item):
        self.__list.insert(item)
        self.__length += 1

    def dequeue(self):
        x = self.__list.findtail()
        self.__list.delete(x, x.key)
        self.__length -= 1
        return x.value

    def count(self):
        self.__length()

    def all(self):
        return self.__list.all()

class StackUsingList:
    '''
    使用链表构造的栈
    '''
    def __init__(self):
        self.__list = List()
        self.__length = 0

    def push(self, item):
        self.__list.insert(item)
        self.__length += 1

    def pop(self):
        x = self.__list.head
        self.__list.delete(x, x.key)
        self.__length -= 1
        return x.value

    def count(self):
        self.__length()

    def all(self):
        return self.__list.all()

class BTreeNode:
    '''
    二叉树结点
    '''
    def __init__(self, left, right, index, \
            key , leftindex, rightindex):
        '''

        二叉树结点

        Args
        ===
        `left` : BTreeNode : 左儿子结点

        `right`  : BTreeNode : 右儿子结点

        `index` : 结点自身索引值

        `key` : 结点自身键值

        `leftindex` : 左儿子结点索引值

        `rightindex` : 右儿子结点索引值

        '''
        self.leftindex = leftindex
        self.rightindex = rightindex
        self.left = left
        self.right = right
        self.index = index
        self.key = key

class BinaryTree:
    '''
    二叉树
    '''
    def __init__(self):
        '''
        二叉树
        '''
        self.lastnode = None
        self.root = None
        self.nodes = []

    def addnode(self, leftindex : int, rightindex : int, selfindex : int, selfkey):
        '''
        加入二叉树结点

        Args
        ===
        `leftindex` : 左儿子结点索引值

        `rightindex` : 右儿子结点索引值

        `selfindex` : 结点自身索引值

        `selfkey` : 结点自身键值

        '''
        leftnode = self.findnode(leftindex)
        rightnode = self.findnode(rightindex)
        x = BTreeNode(leftnode, rightnode, selfindex, \
            selfkey, leftindex, rightindex)
        self.nodes.append(x)
        self.lastnode = x
        return x
        
    def renewall(self) -> None:
        '''
        更新/连接/构造二叉树
        '''
        for node in self.nodes:
            node.left = self.findnode(node.leftindex)
            node.right = self.findnode(node.rightindex)
    
    def findleftrightnode(self, node : BTreeNode) -> list:
        '''
        找出二叉树某结点的所有子结点

        Args
        ===
        `node` : BTreeNode : 某结点
        '''
        array = []
        if node != None:
            # 递归找到左儿子所有的结点
            leftnodes = self.findleftrightnode(node.left)
            # 递归找到右兄弟所有的结点
            rightnodes = self.findleftrightnode(node.right)
            if leftnodes != None and len(leftnodes) != 0:
                # 连接两个集合
                array = array + leftnodes
            if rightnodes != None and len(rightnodes) != 0:
                # 连接两个集合
                array = array + rightnodes
            # 将自己本身的结点也加入集合
            array.append({ "index":node.index, "key" : node.key})
            if len(array) == 0:
                return None
            return array
        return None

    def all(self) -> list:
        '''
        返回二叉树中所有结点索引值，键值构成的集合
        '''
        array = []
        for node in self.nodes:
            array.append({ "index":node.index,"key" : node.key})
        return array

    def keys(self) -> list:
        '''
        返回二叉树中所有结点键值构成的集合
        '''
        array = []
        for node in self.nodes:
            array.append(node.key)
        return array

    def findnode(self, index : int):
        '''
        根据索引寻找结点`O(n)`

        Args
        ===
        `index` : 索引值
        '''
        if index == None:
            return None
        for node in self.nodes:
            if node.index == index:
                return node
        return None
