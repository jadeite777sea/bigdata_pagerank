#!/usr/bin/env python
# coding: utf-8

# In[173]:


from collections import defaultdict
import struct
from sys import getsizeof
import numpy as np
import time
import scipy
import networkx as nx


# In[174]:


def data_solve(file, data):
    m = defaultdict(list)
    n = defaultdict(int)
    #得到节点出度，以及知道目的节点由哪些节点指向
    with open(file, 'r') as f:
        for i in f:
            a,b = [int(x) for x in i.split()]
            n[a]+= 1
            m[b].append(a)
    #使用二进制读写来加速读取速度
    with open(data,'wb') as f:
        for a,b in sorted(m.items()):
            x=a.to_bytes(2,"little")
            h=len(b)
            x=x+h.to_bytes(2,"little")
            for i in b:
                x=x+i.to_bytes(2,"little")
            f.write(x)
    return n


# In[175]:


def data_decode(data):
    m = defaultdict(list)
    with open(data,'rb') as f:
        #文件指针
        i=0
        k=f.read()
        while i!=len(k):
            x=int.from_bytes(k[i:i+2], byteorder='little', signed=False)
            i+=2
            h=int.from_bytes(k[i:i+2], byteorder='little', signed=False)
            i+=2
            for j in range(h):
                m[x].append(int.from_bytes(k[i:i+2], byteorder='little', signed=False))
                i+=2
    return m


# In[177]:


def base_pagerank(file,data,belta,error,max_iteration):
    nodes=data_solve(file, 'data.bin')
    M=data_decode('data.bin')
    #只考虑有出度或是入度的节点
    s=set(sorted(sorted(M)+sorted(nodes)))
    sl=list(s)
    idx={}
    for i in range(len(sl)):
        idx[sl[i]]=i
    N=len(set(sorted(M)+sorted(nodes)))
    start = time.time()
    #使用Teleport解决Dead Ends问题
    a=np.zeros(N)
    for j,i in zip(range(N),s):
            if nodes[i]==0:
                a[j]=1/N
    r=np.ones(N)/N
    r_new=np.ones(N)
    temp2=(1-belta)/N
        #迭代直至收敛
    iteration=0
    while True:
        iteration+=1
        for i,h in zip(s,range(N)):
            m=np.zeros(N)
            temp1=np.dot(a,r)
            if len(M[i])!=0:
                for j in M[i]:
                    temp1+=r[idx[j]]/nodes[j]
            temp1=belta*temp1
            r_new[h]=temp1+temp2
        if sum(abs(r_new-r))<error or iteration==max_iteration:
            break
        r=r_new.copy()
    end = time.time()
    print(iteration)
    print('time: {}s.'.format(end - start))
    index=np.argsort(r_new)[::-1][:100]
    c_index=[sl[i] for i in index]
    score=np.sort(r_new)[::-1][:100]
    return c_index,score
        


# In[131]:



def networkx_pagerank(file):
    G = nx.DiGraph()
    index=list()
    score=list()
    with open(file) as f:
        for i in f:
            h,t = [int(x) for x in i.split()]
            G.add_edge(h,t)
    pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
    i=0
    for node, value in sorted(pr.items(), key=lambda x: x[1],
                                  reverse=True):
        i+=1
        index.append(node)
        score.append(value)
        if i==100:
            break
    return index,score
        


# In[179]:


def write_back_result(c_index,score,name):
    with open(name,'w') as f:
        for index,scorei in zip(c_index,score):
            f.write("{}\t{}\n".format(index,scorei))


# In[180]:


if __name__ == "__main__":
    c_index,score=base_pagerank('Data.txt', 'data.bin',0.85,1e-10,1000)
    write_back_result(c_index,score,"base_result.txt")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




