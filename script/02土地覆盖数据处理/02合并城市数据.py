# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 17:41:39 2021

@author: Xhpan
"""

from osgeo import gdal
import numpy as np
import pandas as pd
import xarray as xr
import os

def WriteTiff(im_data, inputdir, path):
    raster = gdal.Open(inputdir)
    im_width = raster.RasterXSize  # 栅格矩阵的列数
    im_height = raster.RasterYSize  # 栅格矩阵的行数
    im_bands = raster.RasterCount  # 波段数
    im_geotrans = raster.GetGeoTransform()  # 获取仿射矩阵信息
    im_proj = raster.GetProjection()  # 获取投影信息

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

path_lulc = "../../origindata/lulc_clcd"
path_urban = "../../origindata/urban/urban2"
raster = gdal.Open(path_urban + "/urban1992.tif")
im_width = raster.RasterXSize #栅格矩阵的列数
im_height = raster.RasterYSize #栅格矩阵的行数
im_bands = raster.RasterCount #波段数
im_geotrans = raster.GetGeoTransform()#获取仿射矩阵信息
im_proj = raster.GetProjection()#获取投影信息
inputPathFiles, inputNames = getTiffFileName(path_lulc, ".tif")


for name in inputNames:
    year = name[-4:]
    lulc_arr = xr.open_rasterio(path_lulc + "/lulc{}.tif".format(str(year))).data[0,:,:]
    urban_arr = xr.open_rasterio(path_urban + "/urban" + str(year) + ".tif").data[0,:,:]

    lulc_arr1 = lulc_arr + urban_arr * 10
    lulc_arr1[np.where(lulc_arr1 > 10)] = 10
    lulc_arr1[np.where(lulc_arr == 15)] = 0
    
    ResultPath = "../../origindata/lulc" + "/lulc" + str(year) +".tif"
    WriteTiff(lulc_arr1, path_lulc + "/lulc{}.tif".format(str(year)), ResultPath)
    
    print(year)