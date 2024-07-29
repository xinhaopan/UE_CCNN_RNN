#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random 
random.seed(0)
 

dataset_path = "../traindata1985" # 路径
segfilepath= dataset_path + '/Labels' # 标签数据路径
saveBasePath= dataset_path + "/Segmentation" # 存放输出txt的路径
 
if not os.path.exists(saveBasePath):
	os.makedirs(saveBasePath)
    
train_percent= 0.8 # tarin：trainval数据 = 9 ： 1

temp_seg = os.listdir(segfilepath)
total_seg = []
for seg in temp_seg:
    if seg.endswith(".tif"):
        total_seg.append(seg)
        
num=len(total_seg)  
list=range(num) 
tr=int(num *train_percent) 
trainval= random.sample(list,tr) 
train=random.sample(trainval,tr)  
 
print("train and trainval size",tr)
ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
 
for i in list:  
    name=total_seg[i][:-4]+'\n'   
    if i in train:  
        ftrain.write(name)  
    else:  
        ftrainval.write(name)  
        
ftrain.close()  
ftrainval.close()  


# In[ ]:




