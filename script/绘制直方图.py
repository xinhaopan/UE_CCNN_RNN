# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 21:44:34 2021

@author: Xhpan
"""

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import xarray as xr

arr = xr.open_rasterio(r"D:\Work\doing\CNN_RNN\data\beijing\Unet_result\Suitability_1_2015_lb.tif").data[0,:,:]


num = []
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if arr[i][j] > 0.1: 
            num.append(arr[i][j])


font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  

group = []
for n in range(0,101):
    group.append(n/100)
    


plt.hist(num, group, histtype='bar', rwidth=0.8)

# plt.legend()

plt.xlabel('')
plt.ylabel('')

plt.title(u'9:1,大于0.1', FontProperties=font)

plt.show()


num = []
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if arr[i][j] > 0.2: 
            num.append(arr[i][j])


font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  

group = []
for n in range(0,101):
    group.append(n/100)
    


plt.hist(num, group, histtype='bar', rwidth=0.8)

# plt.legend()

plt.xlabel('')
plt.ylabel('')

plt.title(u'9:1,大于0.2', FontProperties=font)

plt.show()

# ----------------------------------------------------------------------------------



num = []
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if arr[i][j] >= 0: 
            num.append(arr[i][j])


font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  

group = []
for n in range(0,101):
    group.append(n/100)
    


plt.hist(num, group, histtype='bar', rwidth=0.8)

# plt.legend()

plt.xlabel('')
plt.ylabel('')

plt.title(u'9:1,全部', FontProperties=font)

plt.show()
#---------------------------------------------------------------------------------------
num = []
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if arr[i][j] > 0: 
            num.append(arr[i][j])


font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  

group = []
for n in range(0,101):
    group.append(n/100)
    


plt.hist(num, group, histtype='bar', rwidth=0.8)

# plt.legend()

plt.xlabel('')
plt.ylabel('')

plt.title(u'9:1,大于0', FontProperties=font)

plt.show()
#---------------------------------------------------------------------------------------
num = []
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if arr[i][j] >= 0.9 and arr[i][j] < 1: 
            num.append(arr[i][j])


font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  

group = []
for n in range(90,101):
    group.append(n/100)
    


plt.hist(num, group, histtype='bar', rwidth=0.8)

# plt.legend()

plt.xlabel('')
plt.ylabel('')

plt.title(u'9:1，0.9-1', FontProperties=font)

plt.show()
#---------------------------------------------------------------------------------------
num = []
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if arr[i][j] >= 0.98 and arr[i][j] < 1: 
            num.append(arr[i][j])


font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  

group = []
for n in range(98,101):
    group.append(n/100)
    


plt.hist(num, group, histtype='bar', rwidth=0.8)

# plt.legend()

plt.xlabel('')
plt.ylabel('')

plt.title(u'9:1，0.98-1', FontProperties=font)

plt.show()