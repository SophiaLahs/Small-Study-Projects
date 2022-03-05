#!/usr/bin/env python
# coding: utf-8

# In[21]:


import math
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.font_manager
import copy
from copy import deepcopy


# In[38]:


def Ziegenproblem(switch=False):
    '''
    switch: whether contestant switches their choice or not
    '''
    #all boxes:
    boxes = [0,1,2]
    #box with the price:
    win = boxes.pop(np.random.randint(0,3))
    #now boxes only contains losing boxes
    #contestants choice:
    choice = np.random.randint(0,3)
    if switch == False:
        if choice == win:
            return 1
        else:
            return 0
    else:
        #sort out one losing box:
        sortout = boxes.pop(np.random.randint(0,2))
        if sortout == choice:
            return 1
        else: 
            if boxes[0] == choice:
                return 1
            else: 
                return 0


# In[39]:


def ziegenplotfunction(number=10000,switch=False):
    ziegenlist = [Ziegenproblem(switch)]
    for i in range(number-1):
        ziegenlist.append(ziegenlist[-1]+Ziegenproblem(switch))
    return ziegenlist

ziegenrange = []
for i in range(10000):
    ziegenrange.append(i)


# In[40]:


plt.figure(figsize=(8,5))
plt.plot(ziegenrange,ziegenplotfunction(switch=False),'.',color='xkcd:orange',label='stays with choice')
plt.plot(ziegenrange,ziegenplotfunction(switch=True),'.',color='xkcd:wine',label='changes choice')
plt.ylabel(r'$f(N)$')
plt.xlabel('$N$')
plt.legend(loc=(0.548,0.163))#,frameon=False
#plt.legend(loc=(0.444,0.1)) #loc 7,frameon=False
#plt.axis([1.75, 10.25, 1.5, 700])
plt.title('Number of wins after N iterations')
plt.savefig('ziegenproblem.pdf')
plt.show()


# In[41]:


def ziegenplotfunction_percentages(number=10000,switch=False):
    ziegenlist = [Ziegenproblem(switch)]
    percentagelist = copy.deepcopy(ziegenlist)
    for i in range(number-1):
        ziegenlist.append(ziegenlist[-1]+Ziegenproblem(switch))
        percentagelist.append(ziegenlist[-1]/(i+2))
    return percentagelist


# In[43]:


plt.figure(figsize=(8,5))
plt.plot(ziegenrange,ziegenplotfunction_percentages(switch=False),'.',color='xkcd:orange',label='stays with choice')
plt.plot(ziegenrange,ziegenplotfunction_percentages(switch=True),'.',color='xkcd:wine',label='changes choice')
plt.ylabel(r'$f(N)$')
plt.xlabel('$N$')
plt.legend(loc=(0.548,0.163))#,frameon=False
#plt.legend(loc=(0.444,0.1)) #loc 7,frameon=False
plt.axis([0,10000, 0, 1])
plt.title('Winning percentage after N iterations')
plt.savefig('ziegenproblem_percentages.pdf')
plt.show()

