#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
exec('from __future__ import print_function')
import torch as t
import torchvision as tv
from IPython.display import Image
t.backends.mkl.is_available()


# In[2]:


with np.load('prediction-challenge-01-data.npz') as fh:
    data_x = fh['data_x']
    data_y = fh['data_y']#label
    test_x = fh['test_x']
#makes unit vectors with ones on the place of the according number out of the numbers to check the results in the end

# TRAINING DATA: INPUT (x) AND OUTPUT (y)
# 1. INDEX: IMAGE SERIAL NUMBER
# 2. INDEX: COLdata_x2OR CHANNEL
# 3/4. INDEX: PIXEL VALUE
print(data_x.shape, data_x.dtype)
print(data_y.shape, data_y.dtype)

# TEST DATA: INPUT (x) ONLY
print(test_x.shape, test_x.dtype)

plt.imshow(data_x[373, 0])
plt.title(data_y[1415])
plt.show()
#20000 pictures, 1 channel, 28x28 pixels


# In[3]:


shuffled=np.load('shuffler.npy')
data_x=data_x[shuffled]
data_y=data_y[shuffled]


# In[4]:


shuffled#new order in respect to former one


# In[5]:


alltdata=t.from_numpy(data_x)#xdata as torch-tensor
txdata=t.from_numpy(data_x[:-4000,:,:,:])#16000 x data points training set
tvaldata=t.from_numpy(data_x[-4000:,:,:,:]).detach()#4000 x data points validation set
ttestdata=t.from_numpy(test_x,).detach()#test data
tydata = t.from_numpy(np.array([np.eye(10,1,-i).flatten() for i in data_y],dtype=np.float32),)# y data

#normalization of data 
normalizer=tv.transforms.Normalize((txdata.mean(),),(txdata.std(),))
normalizer_val=tv.transforms.Normalize((tvaldata.mean(),),(tvaldata.std(),))
normalizer_test=tv.transforms.Normalize((ttestdata.mean(),),(ttestdata.std(),))
for i,tdat in enumerate(txdata):
    txdata[i]=normalizer(tdat)
for i,tdat in enumerate(tvaldata):
    tvaldata[i]=normalizer_val(tdat)
for i,tdat in enumerate(ttestdata):
    ttestdata[i]=normalizer_test(tdat)

dset=t.utils.data.DataLoader(tuple(zip(txdata,tydata)),shuffle=True,batch_size=5,num_workers=0)#data reshuffled at every epoch


# In[6]:


class Net(nn.Module):

    def __init__(self,ch1,ch2):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.pool = nn.MaxPool2d(2,2)#pooling layer, from each 2x2 square, just the most intense input is being adopted
        self.conv1 = nn.Conv2d(1, ch1, 5)#convolution layer
        self.conv2 = nn.Conv2d(ch1, ch2, 5)#convolution layer
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(ch2 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.pool(F.relu(self.conv1(x)))
        # If the size is a square you can only specify a single number
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net(6,16)
print(net)


# In[7]:


lossf = nn.MSELoss()# mean square error loss
optimizer = optim.Adam(net.parameters(), lr = 0.0003)


# In[11]:


for epoch in range(10):
    losstrack = 0.
    for i, (inp,labels) in enumerate(dset):
        optimizer.zero_grad()
        output = net(inp)
        loss = lossf(output,labels)
        losstrack += loss.item()
        if i%200 == 199:
            print("Epoch:",epoch,"Batch: ",i,"   Loss: ",losstrack/200)
            losstrack = 0.
        loss.backward()
        optimizer.step()


# In[12]:


#How to save model
np.save('shuffler',shuffled)
t.save(net.state_dict(),"bmodel")


# In[8]:


#Loading model
net.load_state_dict(t.load('bmodel'))


# In[9]:


tester = lambda arr,valls:np.sum(abs((arr-np.max(arr,axis=1).reshape(-1,1)==0).astype(float)-valls))


# In[10]:


validation_results=net(tvaldata).detach().numpy()
y_validation=tydata[-4000:].detach().numpy()


# In[11]:


notequal = np.where(np.argmax(validation_results, axis=1) - np.argmax(y_validation, axis=1) != 0)


# In[29]:


notequal


# In[30]:


for i in np.arange(0,80,1):
    plt.imshow(data_x[i, 0])
    plt.title(data_y[i])
    plt.show()


# In[12]:


len(np.argmax(y_validation, axis=1)[notequal])


# In[13]:


len(np.argmax(y_validation, axis=1)[notequal])/4000


# In[14]:


xtest=txdata.detach()


# In[15]:


xresults=net(xtest).detach().numpy()


# ## Calculating Prediction for test data

# In[21]:


testresult=net(ttestdata).detach().numpy()


# In[22]:


np.max(testresult,axis=1).reshape(-1,1)


# In[23]:


resmatrix=((testresult-np.max(testresult,axis=1).reshape(-1,1))==0).astype(float)


# In[24]:


final_prediction=(resmatrix*np.arange(10)).sum(axis=1).astype(int)


# In[20]:


np.save('prediction',final_prediction)


# In[25]:


final_prediction


# In[26]:


prediction = (resmatrix*np.arange(10)).sum(axis=1).astype(int)

# MAKE SURE THAT YOU HAVE THE RIGHT FORMAT
assert prediction.ndim == 1
assert prediction.shape[0] == 2000


# In[ ]:


np.save('prediction.npy', prediction)


# In[28]:


prediction


# In[31]:


plt.imshow(data_x[-2, 0])
plt.title(data_y[-2])
plt.show()


# In[ ]:


#keras: model.summary()

