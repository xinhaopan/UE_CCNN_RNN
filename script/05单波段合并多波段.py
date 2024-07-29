# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 22:20:17 2021

@author: Xhpan
"""

from osgeo import gdal
import xarray as xr
import numpy as np

def WriteTiff(im_data, im_width, im_height, im_bands, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # if len(im_data.shape) == 3:
    #     im_bands, im_height, im_width = im_data.shape
    # elif len(im_data.shape) == 2:
    #     im_data = np.array([im_data])
    # else:
    #     im_bands, (im_height, im_width) = 1, im_data.shape
        # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset
    
def compositeBands(data_dirs,im_bands,ResultPath):
    arr_list = []
    for data_dir in data_dirs:
        data_arr = xr.open_rasterio(data_dir).data[0,:,:]
        arr_list.append(data_arr)
        # print(data_dir)
    all_arr = np.stack(arr_list)
    raster = gdal.Open(data_dir)
    im_width = raster.RasterXSize #栅格矩阵的列数
    im_height = raster.RasterYSize #栅格矩阵的行数
    im_geotrans = raster.GetGeoTransform() #获取仿射矩阵信息
    im_proj = raster.GetProjection() #获取投影信息
    WriteTiff(all_arr, im_width, im_height, im_bands, im_geotrans, im_proj, ResultPath) 
    

im_bands = 11 # 要合并的影像数量
data_path = r"D:\Work\doing\CNN_RNN\data\beijing\inputdata"
# data_dirs = [] # 要合并的影像，通过一个循环写入
names =["ALT","SLO","HID","NAD","PRD","RAD","HSD","DC1","DC2","RD1","RD2"]

L = [1985]
for i in range(1990,2018,5):
    L.append(i)

for year in L:
    data_dirs = [] # 要合并的影像，通过一个循环写入
    for name in names:
        data_dirs.append(data_path+ "/" + name + ".tif")
    data_dirs.append(r"D:\Work\doing\CNN_RNN\data\beijing\inputdata\lulc" + "/lulc" + str(year) + ".tif")
    ResultPath = r"D:\Work\doing\CNN_RNN\data\beijing\inputdata\all11" + "/all" + str(year) + ".tif"
    compositeBands(data_dirs,im_bands,ResultPath)
    print(year)
    
    




