# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 20:51:11 2022

@author: Xhpan
"""
from osgeo import gdal
import numpy as np
import os

def LoadData(filename):
    file = gdal.Open(filename)
    if file == None:
        print(filename + " can't be opened!")
        return
    nb = file.RasterCount

    L = []
    for i in range(1, nb + 1):
        band = file.GetRasterBand(i)
        background = band.GetNoDataValue()
        data = band.ReadAsArray()
        data = data.astype(np.float32)
        index = np.where(data == background)
        data[index] = 0
        L.append(data)
    data = np.stack(L,0)
    if nb == 0:
        data = data[0,:,:]
    
    return data


uppath = os.path.dirname(os.getcwd())

arr_boundary = LoadData(uppath + "/inputdata/boundary/boundary.tif")

startyear = 2015
arr_label= LoadData(uppath +  "/origindata/urban/urban2/urban{}.tif".format(str(startyear)))
arr_label[np.where(arr_boundary == 0)] = np.nan
labeldata = arr_label.reshape(-1)
labeldata = labeldata[~np.isnan(labeldata)]
if not os.path.exists("../LSTMdata/label"):
	os.makedirs("../LSTMdata/label")
np.save("../LSTMdata/label/labeldata.npy",labeldata)

files = []
names =["ALT","SLO","HID","NAD","PRD","RAD","HSD","DC1","DC2","DC3","DC4","RD1","RD2","SCL","SOC","SSD","SST","TEM"] # 18
for name in names:
    files.append(uppath + "/inputdata/" + name + ".tif")

years = [1992,1993,1995,1996,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2012]
for i in years:
    files.append(uppath +  "/inputdata/lulc/lulc{}.tif".format(str(i)))
    

for file in files:
    arr_train= LoadData(file)
    arr_train[np.where(arr_boundary == 0)] = np.nan
    traindata = arr_train.reshape(-1)
    traindata = traindata[~np.isnan(traindata)]
    
    if file == files[0]:
        resultdata = np.copy(traindata)
    else:
        resultdata = np.vstack((resultdata,traindata))
    print(file)
    if not os.path.exists("../LSTMdata/train"):
    	os.makedirs("../LSTMdata/train")
np.save("../LSTMdata/train/traindata.npy",resultdata)
    


