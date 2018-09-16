
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
    
    def transition_function(self, q, Ti):
        """
        变迁函数d
        """
        return q

    def finite_automaton_matcher(self, T, d, m):
        """
        字符串匹配自动机的简易过程
        """
        n = len(T)
        q = 0
        for i in range(n):
            q = self.transition_function(q, T[i])
            if q == m:
                print('Pattern occurs with shift %d' % (i - m))

    def compute_transition_function(self, P, sigma):
        """
        下列过程根据一个给定模式`P[1..m]`来计算变迁函数`epsilon`, 运行时间为`O(m^3|∑|)`
        """
        m = len(P)
        for q in range(m + 1):
            for a in sigma:
                k = min(m + 1, q + 2)
                while P[k] != P[q]:
                    k -= 1
                epsilon = k
        return epsilon
    
    def kmp_matcher(self, T, P):
        """
        Knuth-Morris-Pratt字符串匹配算法
        """
        pass

_inst = _StringMatch()

native_string_matcher = _inst.native_string_matcher
rabin_karp_matcher = _inst.rabin_karp_matcher
finite_automaton_matcher = _inst.finite_automaton_matcher
compute_transition_function = _inst.compute_transition_function
kmp_matcher = _inst.kmp_matcher

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
