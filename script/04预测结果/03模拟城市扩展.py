# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 14:20:47 2021

@author: Xhpan
"""

from osgeo import gdal
import numpy as np
import xarray as xr
from openpyxl import Workbook

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

def LoadData1(filename):
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
        data[index] = 3
    return data

# 精度评价部分
def getOAKappe(pred_arr, true_arr):
    OA_kappa_arr = pred_arr * 10 + true_arr

    TP = np.size(np.where(OA_kappa_arr == 11))  # 结果、实际都是城市
    FP = np.size(np.where(OA_kappa_arr == 10))  # 结果是城市，实际不是城市
    FN = np.size(np.where(OA_kappa_arr == 1))  # 结果不是，实际是城市
    TN = np.size(np.where(OA_kappa_arr == 0))  # 结果，实际都不是城市
    TotalNum = TP + FP + FN + TN
    OverAccury = (TP + TN) / TotalNum
    kappa = (float((TP + TN) * TotalNum) - float(((FN + TN) * (FP + TN) + (FN + TP) * (FP + TP)))) / (
            float(TotalNum * TotalNum) - float(((FN + TN) * (FP + TN) + (FN + TP) * (FP + TP))))
    return OverAccury, kappa


def getFOM(true_arr, pred_arr, startYear_arr):
    changeTrue = true_arr - startYear_arr
    changePred = pred_arr - startYear_arr
    A = np.sum(np.where((changeTrue == 1) & (changePred == 0)))
    B = np.sum(np.where((changeTrue == 1) & (changePred == 1)))
    C = np.sum(np.where((changeTrue == 0) & (changePred == 1)))
    return B / (A + B + C)

endyear = 1990
beginyear = 1995
ResultPath = r"D:\Work\doing\CNN_RNN\data\beijing\simdata19901"
path = r"D:\Work\doing\CNN_RNN\data\beijing\origindata\urban\1985_2017_2"

boundary_dir = r"D:\Work\doing\CNN_RNN\data\beijing\inputdata\boundary.tif" # 研究区范围，1为研究区，0为非研究区
boundary_arr = xr.open_rasterio(boundary_dir).data[0,:,:]

nei_arr = xr.open_rasterio(ResultPath + "/NeighborEffect.tif").data[0,:,:]
sui_arr = xr.open_rasterio(ResultPath + "/simdata2000\Suitability1.tif").data[0,:,:]

# prability_arr = nei_arr * sui_arr
sui_arr = (sui_arr-sui_arr.min())/(sui_arr.max()-sui_arr.min())*100
prability_arr = nei_arr + sui_arr


urban_arr = xr.open_rasterio(r"D:\Work\doing\CNN_RNN\data\beijing\origindata\urban\1985_2017_2\urban" + str(endyear) + ".tif").data[0,:,:]
endYearCount = np.sum(urban_arr == 1) # 模拟未来则直接输入未来的城市像元数量
urbanOld_arr = xr.open_rasterio(r"D:\Work\doing\CNN_RNN\data\beijing\origindata\urban\1985_2017_2\urban" + str(beginyear) + ".tif").data[0,:,:]
PiexlNumber = endYearCount - np.sum(urbanOld_arr == 1)

raster = gdal.Open(r"D:\Work\doing\CNN_RNN\data\beijing\origindata\urban\1985_2017_2\urban" + str(endyear) + ".tif")
nRows = raster.RasterYSize  # 行数
nCols = raster.RasterXSize  # 列数

im_width = raster.RasterXSize  # 栅格矩阵的列数
im_height = raster.RasterYSize  # 栅格矩阵的行数
im_bands = raster.RasterCount  # 波段数
im_geotrans = raster.GetGeoTransform()  # 获取仿射矩阵信息
im_proj = raster.GetProjection()  # 获取投影信息
prability_arr[np.where(urbanOld_arr == 1)] = 0
prability_arr[np.where(boundary_arr == 0)] = 0

ResultFile = ResultPath + "/" + "Probability_arr" + ".tif"
WriteTiff(prability_arr, im_width, im_height, im_bands, im_geotrans, im_proj, ResultFile)

PkSort_R_Index = np.argsort(-prability_arr)  #降序索引
Pk = prability_arr.reshape(-1)
urbanOld_arr = urbanOld_arr.reshape(-1)
PkSort_R_Index = np.argsort(-Pk)  #降序索引
urbanOld_arr[PkSort_R_Index[0:PiexlNumber]] = 1  #提取概率最高的赋值为城市[0:PkSort_R_Index]] = 1  #提取概率最高的赋值为城市

urbanOld_arr = np.array(urbanOld_arr).reshape([nRows, nCols])
ResultFile = ResultPath + "/" + "simurban" + str(endyear)  + ".tif"
WriteTiff(urbanOld_arr, im_width, im_height, im_bands, im_geotrans, im_proj, ResultFile)

# 精度评价
startYear_arr = LoadData(path + "/urban" + str(beginyear) + ".tif")
pred_arr = urbanOld_arr
true_arr = LoadData(path + "/urban" + str(endyear) + ".tif")
use_arr = LoadData1(r"D:\Work\doing\CNN_RNN\data\beijing\inputdata\ALT.tif")

startYear_arr[np.where(use_arr == 3)] = 3
pred_arr[np.where(use_arr == 3)] = 3
true_arr[np.where(use_arr == 3)] = 3

OverAccury, kappa = getOAKappe(pred_arr, true_arr)
Fom = getFOM(true_arr, pred_arr, startYear_arr)
print("OverAccury,kappa,Fom:" + str(OverAccury) + " " + str(kappa) + " " + str(Fom))


workbook = Workbook()
save_file = path + "/" + "unet精度.xlsx"
worksheet1 = workbook.active 
worksheet1.append(["OverAccury","Kappa","FOM"])
 
worksheet1.append([OverAccury,kappa,Fom])
workbook.save(filename=save_file)




