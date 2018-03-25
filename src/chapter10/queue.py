
class Queue:

    def __init__(self, iterable = None):
        self.array = []
        if iterable != None:
            self.array = list(iterable)

    def enqueue(self, item):
        self.array.append(item)

    def head(self):
        return -1

    def tail(self):
        return -1

    def count(self):
        return len(self.array)
