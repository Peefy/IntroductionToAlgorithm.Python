---
---
layout: post
title: "用Python实现的数论相关算法"
description: "用Python实现的数论相关算法"
categories: [Python]
tags: [python]
redirect_from:
  - /2018/09/11
---


## 数论相关算法

```python


import random as _rand

class _NumberTheory:
    """
    数论相关算法集合
    """
    def __init__(self):
        """
        数论相关算法集合
        """
        pass

    def gcd(self, a : int, b : int):
        """
        Summary
        ====
        求两个数的最大公约数

        Args
        ===
        `a`: 数字1

        `b`: 数字2

        Return
        ===
        `num` : 最大公约数

        Example
        ===
        ```python
        >>> gcd(24, 30)
        >>> 6
        ```

        """
        assert a >= 0 and b >= 0
        if a == 0 and b == 0:
            return 0
        return self.euclid(a, b)

    def euclid(self, a, b):
        """
        欧几里得算法
        """
        if b == 0:
            return a
        return self.euclid(b, a % b)

    def extend_euclid(self, a, b):
        """
        推广欧几里得算法
        """
        if b == 0:
            return (a, 1, 0)
        (d_ , x_, y_) = self.extend_euclid(b, a % b)
        d, x, y = d_, y_, x_ - (a // b) * y_
        return (d, x, y)

    def ismutualprime(self, a : int, b : int):
        """
        判断两个数是不是互质数
        Args
        ===
        `a`: 数字1

        `b`: 数字2
        """
        return self.gcd(a, b) == 1

    def modular_linear_equation_solver(self, a, b, n):
        """
        求模线性方程组
        """ 
        d, x, y = self.extend_euclid(a, n)
        if d or b:
            x0 = x * (b / d) % n
            for i in range(d):
                print((x0 + i * (n / d)) % n)
        else:
            print('no solotion')

    def modular_exponentiation(self, a, b, n):
        """
        运用反复平方法求数的幂
        """
        c = 0
        d = 1
        bit = bin(b)
        bit = bit[2::]
        bit_list = [int(c) for c in bit]
        d_list = []
        for b in bit_list:
            c = 2 * c
            d = (d * d) % n
            if b == 1:
                c += 1
                d = (d * a) % n
        return d
    
    def witness(self, a, n):
        """
        WIRNESS测试函数
        """
        bit_str = bin(n - 1)
        t = 0
        length = len(bit_str)
        for i in range(length - 1, -1, -1):
            if bit_str[i] == '0':
                t += 1
            else:
                break
        bit_str = bit_str[0:length - t]
        u = int(bit_str, 2)
        x = [0] * (t + 1)
        x[0] = self.modular_exponentiation(a, u, n)
        for i in range(1, t + 1):
            x[i] = (x[i - 1] ** 2) % n
            if x[i] == 1 and x[i - 1] != 1 and x[i - 1] != (n - 1):
                return True
        if x[t] != 1:
            return True
        return False

    def miller_rabin(self, n, s):
        """
        Miller-Rabin随机性素数测试方法
        """
        for j in range(1, s + 1):
            a = _rand.randint(1, n - 1)
            if self.witness(a, n):
                return "Composite"
        return "Prime"

    def pollard_rho(self, n):
        """
        整数的因子分解 Pollard的rho启发式方法

        Args
        ===
        `n` : 被分解的数字

        """
        i = 1
        x = _rand.randint(0, n - 1)
        y = x
        k = 2
        while True:
            i += 1
            x = (x ** 2 - 1) % n
            d = self.gcd(y - x, n)
            if d != 1 and d != n:
                print(d)
            if i == k:
                y = x
                k = 2 * k

__number_theory_instance = _NumberTheory()

gcd = __number_theory_instance.gcd
euclid = __number_theory_instance.euclid
extend_euclid = __number_theory_instance.extend_euclid
ismutualprime = __number_theory_instance.ismutualprime
modular_linear_equation_solver = __number_theory_instance.modular_linear_equation_solver
modular_exponentiation = __number_theory_instance.modular_exponentiation
miller_rabin = __number_theory_instance.miller_rabin
pollard_rho = __number_theory_instance.pollard_rho

def test():
    """
    测试函数
    """
    print(gcd(24, 30))
    print(euclid(24, 30))
    print(extend_euclid(24, 30))
    print(gcd(24, 30))
    print(modular_linear_equation_solver(14, 30, 100))
    print(modular_exponentiation(7, 560, 561))
    print(miller_rabin(561, 10))
    print(pollard_rho(12))

if __name__ == '__main__':
    test()
else:
    pass


```

[Github Code](https://github.com/Peefy/IntroductionToAlgorithm.Python/blob/master/src/chapter31)
