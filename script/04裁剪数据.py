# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 19:46:25 2021

@author: Xhpan
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 12:49:29 2021
同时裁剪标签和图像为指定大小
只保留有城市部分且不包含研究区外的部分
@author: Xhpan
"""
from osgeo import gdal
import xarray as xr
import numpy as np
import os

def Boundary_Check(x,y,dis,bou):
    x_min = x - dis
    x_max = x + dis + 1
    y_min = y - dis
    y_max = y + dis + 1
    if (x_min < 0) | (y_min < 0) | (x_max > bou[0]) | (y_max > bou[1]):
        return(False)
    else:
        return(True)
        
def get_extent(fn):
    '''Returns min_x, max_y, max_x, min_y'''
    ds = gdal.Open(fn)
    gt = ds.GetGeoTransform()
    return (gt[0], gt[3], gt[0] + gt[1] * ds.RasterXSize,
            gt[3] + gt[5] * ds.RasterYSize)
    

def mkdir(path):
    # 引入模块
    import os
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        #print(path +' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        #print(path +' 目录已存在')
        return False
    
def calTransform(ori_transform,offset_x,offset_y):
    # 读取原图仿射变换参数值
    top_left_x = ori_transform[0]  # 左上角x坐标
    w_e_pixel_resolution = ori_transform[1] # 东西方向像素分辨率
    top_left_y = ori_transform[3] # 左上角y坐标
    n_s_pixel_resolution = ori_transform[5] # 南北方向像素分辨率
    # 根据反射变换参数计算新图的原点坐标
    top_left_x = top_left_x + offset_x * w_e_pixel_resolution
    top_left_y = top_left_y + offset_y * n_s_pixel_resolution
    # 将计算后的值组装为一个元组，以方便设置
    dst_transform = (top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4], ori_transform[5])
    return dst_transform
    
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



def cutdata(image_size,label_dir,cutdata_dir,boundary_dir,resultpath,pixarr_save):
    print("begin")
    urban_arr = xr.open_rasterio(label_dir).data[0,:,:]
    # cutdata__arr = xr.open_rasterio(cutdata_dir).data[0,:,:]
    boundary_arr = xr.open_rasterio(boundary_dir).data[0,:,:]
    Boundary = (boundary_arr.shape)
    
    CAdata_ds = gdal.Open(cutdata_dir)
    band_num = CAdata_ds.RasterCount
    ori_transform = CAdata_ds.GetGeoTransform()
    driver = gdal.GetDriverByName('gtiff')
    pix_list = []
    i = 0
    num = Boundary[0]//100
    
    # 裁剪标签数据
    label_ds = gdal.Open(label_dir)
    
    for x in range(0,Boundary[0],image_size):
        for y in range(0,Boundary[1],image_size):
            if Boundary_Check(x, y, image_size, Boundary): # 检查是否超出边界
                rang_arr = boundary_arr[x:x+image_size,y:y+image_size] # 检测是否超出研究区边界
                if 0 not in rang_arr:
                    data_arr = urban_arr[x:x+image_size,y:y+image_size] # 检查范围内是否有城市
                    if 1 in data_arr and 2 not in data_arr:
                        pix_list.append([x, y, i])
                        if not os.path.exists(resultpath + "/images"):
                            os.makedirs(resultpath + "/images")
                        Out_name = resultpath + "/images/" + str(i).zfill(5) +  ".tif"
                        out_ds = driver.Create(Out_name, image_size, image_size, band_num, gdal.GDT_Float64)
                        row = int(x)
                        col = int(y)
                        out_ds.SetGeoTransform(calTransform(ori_transform,col,row))
                        for band in range(band_num):
                            out_band = out_ds.GetRasterBand(band + 1)
                            Input_band = CAdata_ds.GetRasterBand(band + 1)
                            Out_arr = Input_band.ReadAsArray(xoff=col, yoff=row, win_xsize=image_size, win_ysize=image_size)
                            out_band.WriteArray(Out_arr)
                            out_band.FlushCache()
                            del out_band, Input_band
                        
                        # 裁剪标签，单波段
                        if not os.path.exists(resultpath + "/labels"):
                            os.makedirs(resultpath + "/labels")
                        Out_name1 = resultpath + "/labels/" + str(i).zfill(5) +  ".tif"
                        out1_ds = driver.Create(Out_name1, image_size, image_size, 1, gdal.GDT_Float64)
                        out1_ds.SetGeoTransform(calTransform(ori_transform,col,row))
                        out1_band = out1_ds.GetRasterBand(1)
                        Input1_band = label_ds.GetRasterBand(1)
                        out1_arr = Input1_band.ReadAsArray(xoff=col, yoff=row, win_xsize=image_size, win_ysize=image_size)
                        out1_band.WriteArray(out1_arr)
                        out1_band.FlushCache()
                        del out1_band, Input1_band
                        
                        i += 1
        if x % num == 0:
            print(" {:.2f}% ".format(x/Boundary[0] * 100), end = "|")
    del CAdata_ds,label_ds
    pix_arr = np.array(pix_list)
    np.save(pixarr_save, arr=pix_arr)
    print("ok")
    
image_size = 27
os.chdir(r"D:\Work\doing\CNN_RNN\data\beijing\inputdata")

startyear = '1985'
endyear = '2000'

urbanpath = r"D:\Work\doing\CNN_RNN\data\beijing\origindata\urban\1985_2017_2\urban"
urban1 = xr.open_rasterio(urbanpath + startyear + ".tif").data[0,:,:]
urban2 = xr.open_rasterio(urbanpath + endyear + ".tif").data[0,:,:]
label_arr = urban1 + urban2 # 0非城市，1城市扩展区域，2城市不变区域

cutdata_dir = "./all/all{}.tif".format(startyear) # 被切的数据
boundary_dir = "./boundary.tif" # 研究区范围，1为研究区，0为非研究区
resultpath = r"D:\Work\doing\CNN_RNN\data\beijing\traindata{}_{}".format(startyear,endyear)
pixarr_save = resultpath + "/pix_list{}_{}.npy".format(startyear,endyear)
label_dir = resultpath + "/urban{}_{}.tif".format(startyear,endyear)
WriteTiff(label_arr,urbanpath + startyear + ".tif",label_dir)

cutdata(image_size,label_dir,cutdata_dir,boundary_dir,resultpath,pixarr_save)





