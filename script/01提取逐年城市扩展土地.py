# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:59:49 2020

@author: Xhpan
"""
from osgeo import gdal
import numpy as np
import os

def LoadData(filename):
    file = gdal.Open(filename)
    if file == None:
       print (filename+" can't be opened!")
       return
    nb = file.RasterCount

    for i in range(1,nb+1):
        band = file.GetRasterBand(i)
        background = band.GetNoDataValue()
        data = band.ReadAsArray()  
        data=data.astype(np.float32)
        index = np.where(data == background)
        data[index] = 0  
    return data

def WriteTiff(im_data,im_width,im_height,im_bands,im_geotrans,im_proj,path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape
        #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset

def getyear(intif,years,outputpath):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    
    for i in range(1,33,years):
        data = LoadData(intif)
        data1 = data
        data1[np.where(data < (33 - i))] = 0
        data1[np.where(data == (33 - i))] = 1
        data1[np.where(data > (33 - i))] = 2
        
        # 输出部分
        raster = gdal.Open(intif)
        im_width = raster.RasterXSize #栅格矩阵的列数
        im_height = raster.RasterYSize #栅格矩阵的行数
        im_bands = raster.RasterCount #波段数
        im_geotrans = raster.GetGeoTransform()#获取仿射矩阵信息
        im_proj = raster.GetProjection()#获取投影信息
        ResultPath = outputpath + "/urban" +  str(1984 + i) + "_" + str(1984 + i + years) + ".tif"
        WriteTiff(data1, im_width, im_height, im_bands, im_geotrans, im_proj, ResultPath) 
        print("urban" + str(1984 + i) + "_" + str(1984 + i + years) + ".tif")

"""
1978,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017

34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1
"""
path = r"D:\Work\doing\CNN_RNN\data\beijing\inputdata\urban" + "/"
intif = r"D:\Work\doing\CNN_RNN\data\beijing\origindata\urban\beijing_urban.tif"

years = 5
outputpath = path + "/fiveyear"
getyear(intif,years,outputpath)

years = 10
outputpath = path + "/tenyear"
getyear(intif,years,outputpath)

years = 15
outputpath = path + "/fifteenyear"
getyear(intif,years,outputpath)

years = 1
outputpath = path + "/oneyear"
getyear(intif,years,outputpath)


        


    
