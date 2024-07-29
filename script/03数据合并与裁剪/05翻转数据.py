# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 16:50:11 2021

@author: Xhpan
"""
import numpy as np
import xarray as xr
import os
from osgeo import gdal
from tqdm import trange


def WriteTiff(im_data,inputfile,outputfile):
    raster = gdal.Open(inputfile)
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
    dataset = driver.Create(outputfile, im_width, im_height, im_bands, datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset
    
def rotarr(filepath):
    filenames = os.listdir(filepath + "/labels")
    for i in trange(len(filenames)):
        filename = filenames[i]
        png = xr.open_rasterio(filepath + "/labels/" + filename).data[0,:,:]
        for rot_num in range(1,4):
            png1 = np.rot90(png, rot_num)
            outputfile = filepath + "/labels/" + filename[:-4] + "_" + str(rot_num) + ".tif"
            WriteTiff(png1,filepath + "/labels/" + filename,outputfile)
            
            jpg = xr.open_rasterio(filepath + "/images/" + filename).data
            new_jpg = []
            for j in jpg:
                new_jpg.append(np.rot90(j, rot_num))
            jpg1 = np.array(new_jpg)
            outputfile = filepath + "/images/" + filename[:-4] + "_" + str(rot_num) + ".tif"
            WriteTiff(jpg1,filepath + "/images/" + filename,outputfile)



filepath = r"D:\Work\doing\CNN_RNN\data\beijing\traindata1985_2000"
rotarr(filepath)






