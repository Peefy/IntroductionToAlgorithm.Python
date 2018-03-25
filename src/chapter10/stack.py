

class Stack:
    '''
    栈
    '''
    def __init__(self, size : int = 0):
        self.array = []
        pass
    
    def isEmpty(self) -> bool: 
        '''
        栈是否为空

        Return
        ===
        `isempty` => bool

        '''
        return False

    @staticmethod
    def newStack(size : int = 0):
        return Stack()