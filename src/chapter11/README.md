
## 散列表 HashTable

散列表(hash table 哈希表)，是根据关键码值(key value)而直接进行访问的数据结构.如python的dict类，C#的Dictionary类，C++的map类，java的HashTable类

通过把关键码值映射到表中一个位置来访问记录，以加快查找的速度

这个函数叫做散列函数，存放记录的数组叫散列表

### ELF Hash

```python

def ELFhash(self, key : str, mod):
        h = 0
        for c in key:
            h = (h << 4) + ord(c)
            g = h & 0xF0000000
            if g != 0:
                h ^= g >> 24;
            h &= ~g;  
        return h // mod  

```
