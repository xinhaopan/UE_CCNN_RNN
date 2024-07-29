# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 13:17:24 2021

@author: Xhpan
"""

import numpy as np
from osgeo import gdal

import xarray as xr
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

def timeSeriesCorrection(filepath,outputpath):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    inputPathFiles, inputNames = getTiffFileName(filepath, ".tif")
    raster = gdal.Open(filepath + "/" + str(inputNames[0]) + ".tif")
    im_width = raster.RasterXSize #栅格矩阵的列数
    im_height = raster.RasterYSize #栅格矩阵的行数
    im_bands = raster.RasterCount #波段数
    im_geotrans = raster.GetGeoTransform()#获取仿射矩阵信息
    im_proj = raster.GetProjection()#获取投影信息
    for i in range(len(inputNames)-1):
        ResultPath = outputpath + "/" + str(inputNames[i + 1]) + ".tif"
        if i == 0:
            arr1 = xr.open_rasterio(filepath + "/" + str(inputNames[i]) + ".tif").data[0,:,:]
            arr2 = xr.open_rasterio(filepath + "/" + str(inputNames[i + 1]) + ".tif").data[0,:,:]
            arr3 = arr1 + arr2
            arr3[np.where(arr3 == 2)] = 1
        else:
            arr1 = arr3
            arr2 = xr.open_rasterio(filepath + "/" + str(inputNames[i + 1]) + ".tif").data[0,:,:]
            arr3 = arr1 + arr2
            arr3[np.where(arr3 == 2)] = 1
        WriteTiff(arr3, im_width, im_height, im_bands, im_geotrans, im_proj, ResultPath)
        print(ResultPath)

filepath =  "../../origindata/urban/urban1"
outputpath = "../../origindata/urban/urban2"
timeSeriesCorrection(filepath,outputpath)



