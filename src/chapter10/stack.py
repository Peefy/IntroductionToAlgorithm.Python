
class Stack:
    '''
    栈
    '''

    __top = -1

    def __init__(self, iterable = None):
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

