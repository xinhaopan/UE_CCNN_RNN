# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:59:49 2020

@author: Xhpan
"""
from osgeo import gdal
import numpy as np

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


path = r"D:\Work\doing\CNN_RNN\data\beijing\origindata\urban" + "/"
intif = path + "beijing_urban.tif"


for i in range(0,33):
    data = LoadData(intif)
    data1 = data
    data1[np.where(data < (33 - i))] = 0
    data1[np.where(data >= (33 - i))] = 1
    
    # 输出部分
    raster = gdal.Open(intif)
    im_width = raster.RasterXSize #栅格矩阵的列数
    im_height = raster.RasterYSize #栅格矩阵的行数
    im_bands = raster.RasterCount #波段数
    im_geotrans = raster.GetGeoTransform()#获取仿射矩阵信息
    im_proj = raster.GetProjection()#获取投影信息
    ResultPath = path + "1985_2017" + "/" + "urban" + str(1985 + i) + ".tif"
    WriteTiff(data1, im_width, im_height, im_bands, im_geotrans, im_proj, ResultPath) 
    print("urban" + str(1985 + i) + ".tif")
    
