# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 12:53:34 2021

@author: Xhpan
"""
from osgeo import gdal 
import xarray as xr

from skimage import morphology as sm
import numpy as np
import os

def WriteTiff(im_data, im_width, im_height, im_bands, im_geotrans, im_proj, path):
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
        im_bands, (im_height, im_width) = 1, im_data.shape
        # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset
    
def getBoundary(filename,urbanID,kernel,ResultPath):
    raster = xr.open_rasterio(filename).data[0,:,:]
    index1 = np.where(raster != urbanID)
    index2 = np.where(raster == urbanID)
    raster[index1] = False
    raster[index2] = True
    img_close = sm.closing(raster, kernel)
    img_open = sm.opening(img_close, kernel)
    raster = gdal.Open(filename)
    im_width = raster.RasterXSize #栅格矩阵的列数
    im_height = raster.RasterYSize #栅格矩阵的行数
    im_bands = raster.RasterCount #波段数
    im_geotrans = raster.GetGeoTransform()#获取仿射矩阵信息
    im_proj = raster.GetProjection()#获取投影信息
    WriteTiff(img_open, im_width, im_height, im_bands, im_geotrans, im_proj, ResultPath)
    
# 获取某目录下所有tif文件
def getTiffFileName(filepath, suffix):
    L1 = []
    L2 = []
    for root, dirs, files in os.walk(filepath):  # 遍历该文件夹
        for file in files:  # 遍历刚获得的文件名files
            (filename, extension) = os.path.splitext(file)  # 将文件名拆分为文件名与后缀
            if (extension == suffix):  # 判断该后缀是否为.c文件
                L1.append(filepath + "/" + file)
                L2.append(filename)
    return L1, L2

urbanID = 1
filepath =  r"D:\Work\doing\CNN_RNN\data\beijing\origindata\urban\1985_2017"
kernel = sm.octagon(2, 1)
inputPathFiles, inputNames = getTiffFileName(filepath, ".tif")

for name in inputNames:
    filename = filepath + "/" + name + ".tif"
    ResultPath = r"D:\Work\doing\CNN_RNN\data\beijing\origindata\urban\1985_2017_1" + "/" + name + ".tif"
    getBoundary(filename,urbanID,kernel,ResultPath)
    print(filename)


