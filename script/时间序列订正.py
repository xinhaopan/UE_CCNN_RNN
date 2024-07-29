# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 13:17:24 2021

@author: Xhpan
"""

import numpy as np
from osgeo import gdal

import xarray as xr

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

years = [1975,1980,1990,2000,2010,2020]

for year in years:
    raster = gdal.Open(r"D:\Work\done\20210924-城市扩展动画-pxh\data\目视判读\urban" + str(year) + ".tif")
    im_width = raster.RasterXSize #栅格矩阵的列数
    im_height = raster.RasterYSize #栅格矩阵的行数
    im_bands = raster.RasterCount #波段数
    im_geotrans = raster.GetGeoTransform()#获取仿射矩阵信息
    im_proj = raster.GetProjection()#获取投影信息
    
    arr = xr.open_rasterio(r"D:\Work\done\20210924-城市扩展动画-pxh\data\目视判读\urban" + str(year) + ".tif").data[0,:,:]
    arr[np.where(arr != arr[0][0])] = 1
    arr[np.where(arr == arr[0][0])] = 0
    
    ResultPath = r"D:\Work\done\20210924-城市扩展动画-pxh\data\目视判读\urban1\urban" + str(year) + ".tif"
    WriteTiff(arr, im_width, im_height, im_bands, im_geotrans, im_proj, ResultPath) 

