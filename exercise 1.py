#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


PMF1=np.asarray([4.0,1.0,6.0,1.0,3.0,1.0,5.0,1.0])
PMF1=PMF1/np.sum(PMF1)
PMF2=np.asarray([2.0,3.0,2.0,5.0,5.0,2.0,3.0,2.0])
PMF2=PMF2/np.sum(PMF2)


# In[3]:


#task1
D1= np.random.choice(8,1000,p=PMF1)
D2= np.random.choice(8,500,p=PMF2)


# In[4]:


#task2
def compareHIST(D,p):
    plt.hist(D,bins=8,range=(0,8),density=True)
    plt.plot(range(0,8),p)
    plt.show()


# In[5]:


compareHIST(D1,PMF1)
compareHIST(D2,PMF2)


# In[6]:


#task3
D3=np.concatenate((D1,D2))
PMF3=2/3*PMF1+1/3*PMF2
compareHIST(D3,PMF3)


# In[7]:


#task5
Labels=np.ones([1500])
Labels[1000:]=2
Dataset=np.stack((D3,Labels))
Dataset=Dataset[:,np.random.permutation(1500)]
d1=Dataset[0,Dataset[1,:]==1]
d2=Dataset[0,Dataset[1,:]==2]


# In[8]:


compareHIST(d1,PMF1)
compareHIST(d2,PMF2)


# In[9]:


#task6
for i in range(0,8):
    CP=[] 
    for j in range(0,8):
        conditional_prob=2/3*(np.count_nonzero(d1==j)/1000)/(2/3*np.count_nonzero(d1==j)/1000+1/3*np.count_nonzero(d2==j)/500)
        CP.append(conditional_prob)
    plt.bar(np.arange(8),CP)


# In[10]:


#task7
#f(x)=7*x
steps=[10,20,40,80,160,320,630,1280,2560,5120,10240]
for i in range(0,11):
    function=[]
    for j in steps:
        random_var=np.random.choice(8,size=j,p=PMF1)
        y=np.mean([random_var[i]*7 for i in range(len(random_var))])
        function.append(y)
    plt.plot(range(0,11),function)
    
#task8
from scipy import stats

mean=np.sum(PMF1*(np.arange(8)*7))
var=np.sum(49*PMF1*(np.arange(8)**2))-mean**2
stdev=var**0.5

cap=[]
floor=[]
for i in steps:
    interval=stats.norm.interval(0.99,loc=mean,scale=stdev/i**0.5)
    cap.append(interval[1])
    floor.append(interval[0])
    
x=range(11)
y=np.ones((len(x),1))*mean
plt.plot(x,cap,color='black',linestyle='--')
plt.plot(x,floor,color='black',linestyle='--')
plt.plot(x,y,color='black',linestyle='-')
plt.show()

