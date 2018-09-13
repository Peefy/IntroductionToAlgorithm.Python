
import random as _rand

class _StringMatch:
    """
    字符串匹配相关算法
    """
    def __init__(self):
        """
        字符串匹配相关算法
        """
        pass

    def native_string_matcher(self, T : str, P : str):
        """
        朴素字符串匹配
        Args
        ===
        `T` : str
        `P` : str
        """
        n = len(T)
        m = len(P)
        if n < m:
            self.native_string_matcher(P, T)
        for s in range(n - m + 1):
            if P[0:m] == T[s:s + m]:
                print('Pattern occurs with shift %d' % s)
    
_inst = _StringMatch()
native_string_matcher = _inst.native_string_matcher

def test():
    """
    测试函数
    """
    native_string_matcher('eeabaaee', 'abaa')
    native_string_matcher('abc', 'dccabcd')

if __name__ == '__main__':
    test()
else:
    pass
