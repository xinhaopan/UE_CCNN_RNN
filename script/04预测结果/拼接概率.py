# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:49:12 2021

@author: Xhpan
"""

import rasterio as rio
from osgeo import gdal
import numpy as np
import os
import xarray as xr

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

def get_extent(tiffileList):
    filename = tiffileList
    rioData = rio.open(filename[0])
    left = rioData.bounds[0]
    bottom = rioData.bounds[1]
    right = rioData.bounds[2]
    top = rioData.bounds[3]
    for ifile in filename[1:]:
        rioData = rio.open(ifile)
        left = min(left, rioData.bounds[0])
        bottom = min(bottom, rioData.bounds[1])
        right = max(right, rioData.bounds[2])
        top = max(top, rioData.bounds[3])
    return left, bottom, right, top, filename

def getRowCol(left, bottom, right, top):
    cols = int((right - left) / 30)
    rows = int((top - bottom) / 30)
    return cols, rows

def to0(output_name,criterion_fn):
    arr0 = xr.open_rasterio(criterion_fn).data[0,:,:]
    arr1 = xr.open_rasterio(output_name).data[0,:,:]
    arr1[np.where(arr0 == 1)] = 0
    WriteTiff(arr1,criterion_fn, output_name)

def UnifiedLineNumber(in_fn,criterion_fn,output_name):
    arr = xr.open_rasterio(criterion_fn).data[0,:,:]
    im_data = np.zeros(arr.shape)
    WriteTiff(im_data,criterion_fn, "zero.tif")
    
    tiffileList = ["zero.tif",in_fn]
    left, bottom, right, top, filename = get_extent(tiffileList)
    cols, rows= getRowCol(left, bottom, right, top)
    n_bands = 1
    arr = np.zeros((n_bands, rows, cols), dtype=np.float64)
    # 打开一个tif文件
    in_ds = gdal.Open(filename[0])
    
    # 新建一个tif文件
    driver = gdal.GetDriverByName('gtiff')
    out_ds = driver.Create(output_name, cols, rows,1,gdal.GDT_Float32)
    # 设置tif文件的投影
    out_ds.SetProjection(in_ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    # 设置新tif文件的地理变换
    gt = list(in_ds.GetGeoTransform())
    gt[0], gt[3] = left, top
    out_ds.SetGeoTransform(gt)
    # 对要拼接的影像进行循环读取
    for ifile in tiffileList:
        in_ds = gdal.Open(ifile)
        # 计算新建的tif文件及本次打开的tif文件之间的坐标漂移
        trans = gdal.Transformer(in_ds, out_ds, [])
        # 得到偏移起始点
        success, xyz = trans.TransformPoint(False, 0, 0)
        x = round(xyz[0])
        y = round(xyz[1])
        
        # x, y, z = map(int, xyz)
        # 读取波段信息
        fnBand = in_ds.GetRasterBand(1)
        data = fnBand.ReadAsArray()
    
        # 影像重合部分处理，重合部分取最大值
        xSize = fnBand.XSize
        ySize = fnBand.YSize
        outData = out_band.ReadAsArray(x, y, xSize, ySize)
        data = np.maximum(data, outData)
        out_band.WriteArray(data, x, y)
    del out_band, out_ds
    os.remove("zero.tif")
    to0(output_name,criterion_fn)
    
in_fn = r"D:\Work\doing\CNN_RNN\data\beijing\traindata19851\Suitability.tif"
criterion_fn = r"D:\Work\doing\CNN_RNN\data\beijing\origindata\urban\1985_2017_2\urban1985.tif"
output_name = r"D:\Work\doing\CNN_RNN\data\beijing\traindata19851\Suitability1.tif"
UnifiedLineNumber(in_fn,criterion_fn,output_name)

