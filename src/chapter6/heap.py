
def left(i):
    return int(2 * i + 1)

def right(i):
    return int(2 * i + 2)

def parent(i):
    return (i + 1) // 2 - 1

def heapsize(A):
    return len(A) - 1