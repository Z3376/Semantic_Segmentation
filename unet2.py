
# coding: utf-8

# In[1]:


from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import Conv2DTranspose
from keras.layers import Concatenate


# In[2]:


s = 512


# In[3]:


inpt = Input((s,s,3))


# In[4]:


c11 = Conv2D(32,(3,3),padding="same",activation="relu")(inpt)
p1 = MaxPooling2D((2,2))(c11)
c21 = Conv2D(64,(3,3),padding="same",activation="relu")(p1)
p2 = MaxPooling2D((2,2))(c21)
c31 = Conv2D(128,(3,3),padding="same",activation="relu")(p2)
p3 = MaxPooling2D((2,2))(c31)


# In[5]:


b1 = Conv2D(256,(3,3),padding="same",activation="relu")(p3)


# In[6]:


u1 = UpSampling2D((2,2))(b1)
concat1 = Concatenate()([u1,c31])
c41 = Conv2D(128,(3,3),padding="same",activation="relu")(concat1)
u2 = UpSampling2D((2,2))(c41)
concat2 = Concatenate()([u2,c21])
c51 = Conv2D(64,(3,3),padding="same",activation="relu")(concat2)
u3 = UpSampling2D((2,2))(c51)
concat3 = Concatenate()([u3,c11])
c61 = Conv2D(32,(3,3),padding="same",activation="relu")(concat3)


# In[7]:


outpt = Conv2D(1,(1,1),activation="sigmoid")(c61)


# In[8]:


u_fcn = Model(inpt,outpt)


# In[9]:


u_fcn.summary()


# In[10]:


u_fcn.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


# In[11]:


import cv2
import numpy as np
import os


# In[12]:


data_folder = "/Users/harsh/Downloads/niflr/appy-data/"
mask_folder = "/Users/harsh/Downloads/niflr/appy-mask/"


# In[13]:


ls_data = list((os.popen("ls "+data_folder).read()).split("\n"))
n = len(ls_data)


# In[14]:


data_array = np.zeros((n,s,s,3))
mask_array = np.zeros((n,s,s))


# In[15]:


try:
    for i in range(n):
        data = cv2.imread(data_folder+ls_data[i],1)
        data_array[i] = cv2.resize(data,(s,s))
        mask = cv2.imread(mask_folder+"mask_"+ls_data[i],0)
        mask_array[i] = cv2.resize(mask,(s,s))
except(cv2.error):
    {}


# In[16]:


data_array = data_array/255
mask_array = mask_array/255


# In[17]:


import matplotlib.pyplot as plt


# In[18]:


plt.figure()
plt.imshow(data_array[1,:,:,2],cmap="gray")
plt.figure()
plt.imshow(mask_array[1],cmap="gray")


# In[19]:


mask_array = np.expand_dims(mask_array,axis=-1)


# In[20]:


u_fcn.fit(data_array,mask_array,batch_size=1,epochs=2)


# In[21]:


u_fcn.save_weights("u_fcn4.h5")


# In[22]:


test_image_path = "/Users/harsh/Downloads/niflr/T2-Store-1.jpeg"
#test_image_path = "/Users/harsh/Downloads/niflr/appy.jpg"
#test_image_path = "/Users/harsh/Downloads/niflr/appy-data/IDShot_540x540.jpg"
#test_image_path = "/Users/harsh/Downloads/niflr/appy-data/102440 copy.jpg"


# In[23]:


test_image = cv2.imread(test_image_path,-1)
(o1,o2,o3) = test_image.shape
test_image = cv2.resize(test_image,(s,s))
test_image = test_image/255


# In[24]:


plt.imshow(test_image[:,:,2],cmap="gray")


# In[25]:


test_image = np.expand_dims(test_image,axis=0)


# In[26]:


pred_mask = u_fcn.predict(test_image)


# In[27]:


pred_mask = cv2.resize(pred_mask[0,:,:,0],(o2,o1))


# In[28]:


plt.imshow(pred_mask,cmap="hot")


# In[29]:


pred_mask = pred_mask>0.2


# In[30]:


plt.imshow(pred_mask,cmap="hot")

