# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 13:58:43 2021

@author: Xhpan
"""

import numpy as np
from osgeo import gdal
import os

def WriteTiff(im_data,inputdir, path):
    raster = gdal.Open(inputdir)
    im_width = raster.RasterXSize #栅格矩阵的列数
    im_height = raster.RasterYSize #栅格矩阵的行数
    im_bands = 1
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

year = 2000

path = r"D:\Work\doing\CNN_RNN\data\beijing\simdata" + str(year)
resultpath = path + "/probality"
if not os.path.exists(resultpath):
    os.makedirs(resultpath)
names = os.listdir(path + "/images")
    
for in_fn in names:
    arr = np.load(path + "/np_save/" +  in_fn[:-4] + ".npy")
    im_data = arr[1]
    inputdir = path + "/images/" + in_fn
    resultfile = resultpath + "/" + in_fn
    WriteTiff(im_data,inputdir, resultfile)
