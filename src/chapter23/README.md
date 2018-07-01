

## 最小生成树

最小权值生成树T,不是使T中边的数目最小化，所有生成树的边数恰好都是|V|-1

对图中每一条边(u,v)∈E,都有一个权值w(u,v)表示连接u和v的代价(需要的接线数目)

希望找出一个无回路的子集T∈E,它连接了所有的顶点，且其权值之和w(T)=∑w(u,v)最小

因为T无回路且连接了所有的顶点,所以它必然是一棵树，称为生成树

因为由最小生成树可以"生成"图G

把确定树T的问题称为最小生成树问题

最小生成树问题的两种算法：Kruskal算法和Prim算法

```python



```

[Github Code](https://github.com/Peefy/CLRS_dugu_code-master/blob/master/src/chapter23)
