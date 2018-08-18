
## 并行计算模型,

比较网络是允许同时进行很多比较的一种算法

排序网络总是能对其他输入进行排序的比较网络,比较网络仅由线路和比较器构成

0-1原理认为，对于属于集合{0,1}的每个输入值,排序网络都能正确运行,则对任意输入值,它也能正确运行

要构造有效的排序网络，第一步是构造一个能对任意双调序列(bitonic sequence)进行的比较网络

合并网络就是指能把两个已排序的输入序列合并为一个有序的输出序列的网络

排序网络SORTER[n]运用合并网络，实现对合并排序算法的并行化

```python


```

[Github Code](https://github.com/Peefy/IntroductionToAlgorithm.Python/blob/master/src/chapter27)
