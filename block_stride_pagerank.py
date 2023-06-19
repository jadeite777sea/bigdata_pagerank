#!/usr/bin/env python
# coding: utf-8

# In[44]:


from collections import defaultdict
import struct
from sys import getsizeof
import numpy as np
import time
import copy
import os
import shutil


# In[45]:


#二进文件的数据处理优化
def data_block_stride_solve(file,block_number):
    m = defaultdict(list)
    n = defaultdict(int)
    #得到节点出度，以及知道目的节点由哪些节点指向
    with open(file, 'r') as f:
        for i in f:
            a,b = [int(x) for x in i.split()]
            n[a]+= 1
            m[b].append(a)
    
    m=dict(sorted(m.items()))
    m1=copy.deepcopy(m)
    #每一块的区间宽度
    block_width=int(max(m)/block_number)
    lift=0
    right=-1
    l=list()
    r=list()
    filepath="./block"
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)
        
    for i in range(block_number):
        lift=right+1
        l.append(lift)
        if i!=block_number-1:
            right=right+block_width
        else:
            right=max(m)
        r.append(right)
        block_name="./block/block_"+str(i)
        with open(block_name,'wb') as f:
            for a,b in sorted(m.items()):
                if a>=lift and a<=right:
                    x=a.to_bytes(2,"little")
                    h=len(b)
                    x=x+h.to_bytes(2,"little")
                    for i in b:
                        x=x+i.to_bytes(2,"little")
                    f.write(x)
                    del(m[a])
                else:
                    break
    for i in m1.keys():
        if n[i]==0:
            continue
        
    return n,l,r


# In[46]:


def data_decode(block_index):
    m = defaultdict(list)
    block_name="./block/block_"+str(block_index)
    with open(block_name,'rb') as f:
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


# In[47]:


#文本文件处理
def data_block_stride_solve_t(file,block_number):
    m = defaultdict(list)
    n = defaultdict(int)
    #得到节点出度，以及知道目的节点由哪些节点指向
    with open(file, 'r') as f:
        for i in f:
            a,b = [int(x) for x in i.split()]
            n[a]+= 1
            m[b].append(a)
    
    m=dict(sorted(m.items()))
    m1=copy.deepcopy(m)
    #每一块的区间宽度
    block_width=int(max(m)/block_number)
    lift=0
    right=-1
    l=list()
    r=list()
    filepath="./block_t"
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)
        
    for i in range(block_number):
        lift=right+1
        l.append(lift)
        if i!=block_number-1:
            right=right+block_width
        else:
            right=max(m)
        r.append(right)
        block_name="./block_t/block_"+str(i)
        with open(block_name,'w') as f:
            for a,b in sorted(m.items()):
                if a>=lift and a<=right:
                    f.write("{} {}\n".format(a,' '.join(str(x) for x in b)))
                    del(m[a])
                else:
                    break
    for i in m1.keys():
        if n[i]==0:
            continue
        
    return n,l,r


# In[48]:


def data_decode_t(block_index):
    m = defaultdict(list)
    block_name="./block_t/block_"+str(block_index)
    with open(block_name,'r') as f:
        for i in f:
            a=i.split(' ')
            index=int(a[0])
            for j in a[1:]:
                m[index].append(int(j))
            
    return m


# In[49]:


def block_stride_pagerank(file,belta,error,max_iteration,block_number,data_func):
    nodes,ll,rr=data_func[0]("Data.txt",block_number)
    start = time.time()
    N=len(nodes)
    s=list(sorted(nodes))
    idx={}
    for i in range(len(s)):
        idx[s[i]]=i
    a=np.zeros(N)
    for j,i in zip(range(N),s):
            if nodes[i]==0:
                a[j]=1/N
    r=np.ones(N)/N
    r_new=np.ones(N)
    temp2=(1-belta)/N
    iteration=0
    while True:
        block_index=0
        M=data_func[1](block_index)
        iteration+=1
        for i,h in zip(s,range(N)):
            if i>=ll[block_index] and i<=rr[block_index]:
                m=np.zeros(N)
                temp1=np.dot(a,r)
                if len(M[i])!=0:
                    for j in M[i]:
                        temp1+=r[idx[j]]/nodes[j]
                temp1=belta*temp1
                r_new[h]=temp1+temp2
            else:
                block_index+=1
                M=data_func[1](block_index)
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
    c_index=[s[i] for i in index]
    score=np.sort(r_new)[::-1][:100]
    return c_index,score        


# In[50]:


def write_back_result(c_index,score,name):
    with open(name,'w') as f:
        for index,scorei in zip(c_index,score):
            f.write("{}\t{}\n".format(index,scorei))


# In[58]:


if __name__ == "__main__":
    c_index,score=block_stride_pagerank("Data.txt",0.85,1e-10,1000,100,[data_block_stride_solve,data_decode])
    write_back_result(c_index,score,"block_stride_result.txt")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




