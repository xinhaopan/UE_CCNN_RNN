# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 19:24:04 2021

@author: Xhpan
"""

from osgeo import gdal
import numpy as np
import os

def LoadData(filename):
    file = gdal.Open(filename)
    if file == None:
        print(filename + " can't be opened!")
        return
    nb = file.RasterCount

    for i in range(1, nb + 1):
        band = file.GetRasterBand(i)
        background = band.GetNoDataValue()
        data = band.ReadAsArray()
        data = data.astype(np.float32)
        index = np.where(data == background)
        data[index] = 0
    return data

# 计算蒙特卡罗邻域
def GetNeighborEffect(startyear_filename, WindowSize):
    raster = gdal.Open(startyear_filename)
    urbanData1 = LoadData(startyear_filename)
    if raster == None:
        print(startyear_filename + '文件无法打开')

    nRows = raster.RasterYSize  # 行数
    nCols = raster.RasterXSize  # 列数
    Width = WindowSize // 2
    center = WindowSize // 2

    NeighborEffect = np.zeros([nRows, nCols])  # 创建图层用于存储邻域影响的值

    AffectedCellsWhere = np.argwhere(urbanData1 == 1)  # 存入所有城市像元的位置

    M_Max = 0.0  # 计算邻域空间内所有像元据中心点距离倒数之和
    for i in range(WindowSize):
        for j in range(WindowSize):
            if i * j != center * center:
                M_Max = M_Max + round((1.0 / ((i - center) ** 2 + (j - center) ** 2) ** 0.5), 8)

    # 逐个模拟非城市像元对邻域内城市像元的影响，结果相加
    k = 0
    for j, i in AffectedCellsWhere:  # j是行X，i是列Y
        k += 1
        Value_U = 0.0
        if (j - Width) < 0:  # 如果行小于下界
            XStart = 0
        else:
            XStart = j - Width
        if (j + Width) > (nRows - 1):  # 如果行大于上界
            XEnd = nRows - 1
        else:
            XEnd = j + Width
        if (i - Width) < 0:  # 如果列小于下界
            YStart = 0
        else:
            YStart = i - Width
        if (i + Width) > (nCols - 1):  # 如果列大于上界
            YEnd = nCols - 1
        else:
            YEnd = i + Width

        # 循环逐个计算某城市像元对其邻域内非城市像元的影响
        # m、n是邻域内的非城市像元，i、j是中心的城市像元
        for m in range(YStart, int(YEnd) + 1): # 列
            for n in range(XStart, int(XEnd) + 1): # 行
                if m != i or n != j:  # 当搜索不为自身
                    if urbanData1[n][m] != 1:  # 城市像元为1，这里找非城市像元
                        # 计算该城市像元对邻域内非城市像元的影响
                        PixelEffect = round((1 / ((m - i) ** 2 + (n - j) ** 2) ** 0.5), 8)  # 当前一个像素对单元的影响
                        NeighborEffect[n][m] += PixelEffect  # 邻域内所有像素对单元影响之和

    NeighborEffect = NeighborEffect / M_Max  # 标准化
    NeighborEffect = np.around((NeighborEffect-NeighborEffect.min())/(NeighborEffect.max()-NeighborEffect.min())*100)

    return NeighborEffect  # 返回空间邻域影响图层

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


WindowSize = 5
startyear_filename = r"D:\Work\doing\CNN_RNN\data\beijing\origindata\urban\1985_2017_2\urban2000.tif"
NeighborEffect_arr = GetNeighborEffect(startyear_filename, WindowSize)
resultpath = r"D:\Work\doing\CNN_RNN\data\beijing\simdata2000\NeighborEffect.tif"
WriteTiff(NeighborEffect_arr,startyear_filename, resultpath)