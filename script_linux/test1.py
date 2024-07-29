# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 23:11:46 2021

@author: Xhpan
"""

from osgeo import gdal
import numpy as np
import math
import os
import rasterio as rio
import xarray as xr
from tqdm import tqdm, trange

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
    if nb == 1:
        data = data[0,:,:]
    return data

def WriteTiff(im_data,inputdir, path):
    raster = gdal.Open(inputdir)
    im_width = raster.RasterXSize #栅格矩阵的列数
    im_height = raster.RasterYSize #栅格矩阵的行数
    im_bands = raster.RasterCount #波段数
    im_geotrans = raster.GetGeoTransform()#获取仿射矩阵信息
    im_proj = raster.GetProjection()#获取投影信息

    
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

years = [1993,1995,1996,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2012]
arr0 = LoadData("../origindata/urban/urban3/urban1992.tif")
for year in years:
    inputpath = "../origindata/urban/urban3/urban" + str(year) + ".tif"
    arr1 = LoadData(inputpath)
    arr0 = arr1 + arr0

years = [1992,1993,1995,1996,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2012]
for i in range(1,19):
    arr1[np.where(arr0 == i)] = int(years[-i])

resultpath = "../origindata/urban/urbanAll.tif"
WriteTiff(arr0, inputpath, resultpath)
resultpath = "../origindata/urban/urbanAll1.tif"
WriteTiff(arr1, inputpath, resultpath)

            


