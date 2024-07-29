from osgeo import gdal
import numpy as np
import os
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

startyear = 1992
endyear = 2015
path_city = "../inputdata/boundary/cityBoundary.tif"
arr_city = LoadData(path_city)
path_urban1 = "../origindata/urban/urban2/urban" + str(startyear) + ".tif"
arr_urban1 = LoadData(path_urban1)
path_urban2 = "../origindata/urban/urban2/urban" + str(endyear) + ".tif"
arr_urban2 = LoadData(path_urban2)

df = pd.DataFrame(columns=['ID','Count',"{}".format(str(startyear)),"{}".format((endyear))])
nums = np.unique(arr_city)
for n in nums:
    if n != 0:
        count2 = np.sum(arr_urban2[np.where(arr_city == n)])
        count1 = np.sum(arr_urban1[np.where(arr_city == n)])
        count = count2 - count1
        df = df.append(pd.DataFrame({'ID':[n],'Count':[count],"{}".format(str(startyear)):[count1],"{}".format((endyear)):[count2]}),ignore_index=True)
resultpath = "../cityCount"
if not os.path.exists(resultpath):
    os.makedirs(resultpath)
df.to_excel(resultpath + "/count" + str(startyear) + "_" + str(endyear) + ".xlsx",index=False)