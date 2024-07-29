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
    
def writearr(CAdata_ds,out_ds,band,col,row,image_size):
    out_band = out_ds.GetRasterBand(band + 1)
    Input_band = CAdata_ds.GetRasterBand(1)
    Out_arr = Input_band.ReadAsArray(xoff=col, yoff=row, win_xsize=image_size, win_ysize=image_size)
    out_band.WriteArray(Out_arr)
    out_band.FlushCache()
    del out_band, Input_band
    


def cutdata(image_size,label_dir,cutdata_dirs,boundary_dir,urban_use,resultpath,pixarr_save):
    print("begin")
    urban_arr = xr.open_rasterio(label_dir).data[0,:,:]
    # cutdata__arr = xr.open_rasterio(cutdata_dir).data[0,:,:]
    boundary_arr = xr.open_rasterio(boundary_dir).data[0,:,:]
    Boundary = (boundary_arr.shape)
    
    driver = gdal.GetDriverByName('gtiff')
    pix_list = []
    i = 0
    num = Boundary[0]//100
    band_num = len(cutdata_dirs)
    # 裁剪标签数据
    label_ds = gdal.Open(label_dir)
    ori_transform = label_ds.GetGeoTransform()
    
    for x in range(0,Boundary[0],image_size):
        for y in range(0,Boundary[1],image_size):
            if Boundary_Check(x, y, image_size, Boundary): # 检查是否超出边界
                rang_arr = boundary_arr[x:x+image_size,y:y+image_size] # 检测是否超出研究区边界
                if 0 not in rang_arr:
                    data_arr = urban_arr[x:x+image_size,y:y+image_size] # 检查范围内是否有城市
                    if 1 in data_arr or urban_use != 1:
                        pix_list.append([x, y, i])
                        if not os.path.exists(resultpath + "/images"):
                            os.makedirs(resultpath + "/images")
                        Out_name = resultpath + "/images/" + str(i).zfill(5) +  ".tif"
                        out_ds = driver.Create(Out_name, image_size, image_size, band_num, gdal.GDT_Float64)
                        row = int(x)
                        col = int(y)
                        out_ds.SetGeoTransform(calTransform(ori_transform,col,row))
                        for band in range(len(cutdata_dirs)):
                            CAdata_ds = gdal.Open(cutdata_dirs[band])
                            writearr(CAdata_ds,out_ds,band,col,row,image_size)
                        
                        # 裁剪标签，单波段
                        if not os.path.exists(resultpath + "/labels"):
                            os.makedirs(resultpath + "/labels")
                        Out_name1 = resultpath + "/labels/" + str(i).zfill(5) +  ".tif"
                        out1_ds = driver.Create(Out_name1, image_size, image_size, 1, gdal.GDT_Float64)
                        out1_ds.SetGeoTransform(calTransform(ori_transform,col,row))
                        writearr(label_ds,out1_ds,0,col,row,image_size)
                        
                        i += 1
        if x % num == 0:
            print(" {:.2f}% ".format(x/Boundary[0] * 100), end = "|")
    del CAdata_ds,label_ds
    pix_arr = np.array(pix_list)
    np.save(pixarr_save, arr=pix_arr)
    print("ok")
    
image_size = 27
names =["ALT","SLO","HID","NAD","PRD","RAD","SHD","DC1","DC2","DC3","DC4","RD1","RD2"]
data_path = "D:\Work\doing\CNN_RNN\data\inputdata"
for year in range(1985,2020,5):
    label_dir = r"D:\Work\doing\CNN_RNN\data\inputdata\1985_2017\urban{}.tif".format(str(year)) # 标签数据
    cutdata_dirs = []
    for name in names:
        cutdata_dirs.append(data_path+ "/" + name + ".tif")
    cutdata_dirs.append(r"D:\Work\doing\CNN_RNN\data\inputdata\lulc" + "/lulc" + str(year) + ".tif")
    
    boundary_dir = r"D:\Work\doing\CNN_RNN\data\inputdata\boundary.tif" # 研究区范围，1为研究区，0为非研究区
    urban_use = 1 # 是否考虑只保留包含城市的部分
    resultpath = r"D:\Work\doing\CNN_RNN\data\traindata{}".format(str(year))
    pixarr_save = resultpath + "/pix_list{}.npy".format(str(year))
    print("Begin:" + str(year))
    cutdata(image_size,label_dir,cutdata_dirs,boundary_dir,urban_use,resultpath,pixarr_save)
    