---
---
layout: post
title: "用Python实现的计算几何学算法"
description: "用Python实现的计算几何学算法"
categories: [Python]
tags: [python]
redirect_from:
  - /2018/09/17
---


## 计算几何学算法

```python

class Point:
    """
    点 `(x, y)`
    """
    def __init__(self, x = 0, y = 0):
        """
        点 `(x, y)`
        """
        self.x = x
        self.y = y
    
    def location(self):
        """
        Return
        ===
        (`x`, `y`)
        """
        return self.x, self.y

class _ComputedGeometry:
    """
    计算几何学算法
    """
    def __init__(self):
        """
        计算几何学算法
        """
        pass

    def direction(self, pi : Point, pj : Point, pk : Point):
        """
        Return
        ===
        `(pj - pi) × (pk - pi)`
        """
        xi, yi = pi.location()
        xj, yj = pj.location()
        xk, yk = pk.location()
        return (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)

    def on_segment(self, pi : Point, pj : Point, pk : Point):
        """
        点`pi`是否在线段`pjpk`上
        """
        xi, yi = pi.location()
        xj, yj = pj.location()
        xk, yk = pk.location()
        if (min(xi, xj) <= xk and xk <= max(xi, xj)) and (min(yi, yj) <= yk and yk <= max(yi, yj)):
            return True
        return False

    def segments_intersect(self, p1 : Point, p2 : Point, p3 : Point, p4 : Point):
        """
        判断线段p1p2和p3p4是否相交，相交返回Ture否则返回False
        """
        d1 = self.direction(p3, p4, p1)
        d2 = self.direction(p3, p4, p2)
        d3 = self.direction(p1, p2, p3)
        d4 = self.direction(p1, p2, p4)
        if (d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0) and \
            (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0):
            return True
        elif d1 == 0 and self.on_segment(p3, p4, p1) == True:
            return True
        elif d2 == 0 and self.on_segment(p3, p4, p2) == True:
            return True
        elif d3 == 0 and self.on_segment(p1, p2, p3) == True:
            return True
        elif d4 == 0 and self.on_segment(p1, p2, p4) == True:
            return True
        return False

_inst = _ComputedGeometry()
segments_intersect = _inst.segments_intersect

def test():
    """
    测试函数
    """
    p1 = Point(0, 0)
    p2 = Point(1, 1)
    p3 = Point(1, 0)
    p4 = Point(0, 1)
    print('线段p1p2和线段p3p4是否相交', segments_intersect(p1, p2, p3, p4))
    print('线段p1p3和线段p2p4是否相交', segments_intersect(p1, p3, p2, p4))

if __name__ == '__main__':
    test()
else:
    pass


```

[Github Code](https://github.com/Peefy/IntroductionToAlgorithm.Python/blob/master/src/chapter33)
