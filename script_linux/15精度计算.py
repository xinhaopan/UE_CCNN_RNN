from osgeo import gdal
import numpy as np
import os
from openpyxl import Workbook,load_workbook
import datetime
import pandas as pd

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

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

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

startyear = 2015
endyear = 2020
arr_city = LoadData("../inputdata/boundary/cityBoundary.tif")
df_name = pd.read_excel("../cityCount/cityid_name.xls")

simnames = ["LUSD/验证精度/sim2020_000/simurban2020_0.tif","urbanExpansion/plan1.tif","urbanExpansion/plan2.tif","urbanExpansion/plan3_w5.tif","urbanExpansion/cities/plan3.tif"]

writer = pd.ExcelWriter('../urbanExpansion/分模型分城市模拟精度.xlsx')
city_list = list(np.unique(arr_city))
for n in range(0,len(simnames)):
    df_result = pd.DataFrame(columns=['cityname', "OA", "Kappa", "FOM"])
    for m in range(0,len(city_list)):
        city_id = city_list[m]
        if city_id != 0:
            arr_boundary = LoadData("../inputdata/boundary/boundary.tif")
            pred_path = "../" + simnames[n]
            pred_arr = LoadData(pred_path)
            pred_arr[np.where(arr_boundary == 0)] = np.nan
            end_arr = LoadData("../origindata/urban/urban3/urban{}.tif".format(str(endyear)))
            end_arr[np.where(arr_boundary == 0)] = np.nan

            pred_arr1 = np.copy(pred_arr)
            end_arr1 =np.copy(end_arr)
            pred_arr1[np.where(arr_city != city_id)] = 3
            end_arr1[np.where(arr_city != city_id)] = 3

            OverAccury, kappa = getOAKappe(pred_arr1, end_arr1)
            start_arr = LoadData("../origindata/urban/urban3/urban{}.tif".format(str(startyear)))

            start_arr[np.where(arr_boundary == 0)] = np.nan
            start_arr1 = np.copy(start_arr)
            start_arr1[np.where(arr_city != city_id)] = 3
            Fom = getFOM(end_arr1, pred_arr1, start_arr1)
            start_arr[np.where(arr_boundary == 0)] = 0

            cityname =  df_name[df_name['id'] == city_id]["市"].values[0]

            print(str(n),"city,OverAccury,kappa,Fom:" + str(cityname) + str(OverAccury) + " " + str(kappa) + " " + str(Fom))
            df_result.loc[m] = [cityname,OverAccury,kappa,Fom]

            df_result.to_excel(writer, str(n))
writer.save()



