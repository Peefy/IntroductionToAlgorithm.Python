
import stack

class Queue:

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
    def __init__(self, iterable = None):
        self.__stack1 = stack.Stack()
        self.__stack2 = stack.Stack()
    
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