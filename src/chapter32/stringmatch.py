
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
    
    def rabin_karp_matcher(self, T : str, P : str, d, q):
        """
        Rabin-Karp字符串匹配算法
        """
        n = len(T)
        m = len(P)
        h = d ** (m - 1) % q
        p = 0
        t = 0
        for i in range(0, m):
            p = (d * p + ord(P[i]) - ord('0')) % q
            t = (d * t + ord(T[i]) - ord('0')) % q
        for s in range(0, n - m + 1):
            if p == t:
                if P[0:m] == T[s:s + m]:
                    print('Pattern occurs with shift %d' % s)
            if s < n - m:
                t = (d * (t - (ord(T[s]) - ord('0')) * h) + ord(T[s + m]) - ord('0')) % p
    
_inst = _StringMatch()
native_string_matcher = _inst.native_string_matcher
rabin_karp_matcher = _inst.rabin_karp_matcher

def test():
    """
    测试函数
    """
    native_string_matcher('eeabaaee', 'abaa')
    native_string_matcher('abc', 'dccabcd')
    native_string_matcher('3141592653589793', '26')
    rabin_karp_matcher('3141592653589793', '26', 10, 11)

if __name__ == '__main__':
    test()
else:
    pass
