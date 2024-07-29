# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 14:26:52 2021

@author: Xhpan
"""

from osgeo import gdal
import math
import xarray as xr
import numpy as np

def GetExtent(in_fn):
    ds=gdal.Open(in_fn)
    geotrans=list(ds.GetGeoTransform())
    xsize=ds.RasterXSize 
    ysize=ds.RasterYSize
    min_x=geotrans[0]
    max_y=geotrans[3]
    max_x=geotrans[0]+xsize*geotrans[1]
    min_y=geotrans[3]+ysize*geotrans[5]
    ds=None
    return min_x,max_y,max_x,min_y

def UnifiedLineNumber(in_fn,criterion_fn,output_name):
    in_ds=gdal.Open(criterion_fn)
    geotrans=list(in_ds.GetGeoTransform())
    width=geotrans[1]
    height=geotrans[5]
    
    # 计算输出图像的行列号
    min_x,max_y,max_x,min_y = GetExtent(criterion_fn)
    columns=math.ceil((max_x-min_x)/width)
    rows=math.ceil((max_y-min_y)/(-height))
    in_band=in_ds.GetRasterBand(1)
    
    driver=gdal.GetDriverByName('GTiff')
    out_ds=driver.Create(output_name,columns,rows,1,gdal.GDT_Float32)
    out_ds.SetProjection(in_ds.GetProjection())
    
    # 计算原图像在新图像位置
    min_x1,max_y1,max_x1,min_y1 = GetExtent(in_fn)
    geotrans[0]=min_x1
    geotrans[3]=max_y1
    out_ds.SetGeoTransform(geotrans)
    out_band=out_ds.GetRasterBand(1)
    #定义仿射逆变换
    inv_geotrans=gdal.InvGeoTransform(geotrans)
    
    in_ds=gdal.Open(in_fn)
    in_gt=in_ds.GetGeoTransform()
    #仿射逆变换
    offset=gdal.ApplyGeoTransform(inv_geotrans,in_gt[0],in_gt[3])
    x,y=map(int,offset)
    # print(x,y)
    trans=gdal.Transformer(in_ds,out_ds,[])#in_ds是源栅格，out_ds是目标栅格
    success,xyz=trans.TransformPoint(False,0,0)#计算in_ds中左上角像元对应out_ds中的行列号
    x,y,z=map(int,xyz)
    # print(x,y,z)
    data=in_ds.GetRasterBand(1).ReadAsArray()
    out_band.WriteArray(data,x,y)#x，y是开始写入时左上角像元行列号
    
    criterion_arr = xr.open_rasterio(criterion_fn).data[0,:,:]
    data_arr = xr.open_rasterio(output_name).data[0,:,:]
    data_arr[np.where(criterion_arr == 1)] = 0
    out_band.WriteArray(data_arr,x,y)#x，y是开始写入时左上角像元行列号
    
    del in_ds,out_band,out_ds
    
in_fn = r"D:\Work\doing\CNN_RNN\data\beijing\traindata19851\Suitability.tif"
criterion_fn = r"D:\Work\doing\CNN_RNN\data\beijing\origindata\urban\1985_2017_2\urban1985.tif"
output_name = r"D:\Work\doing\CNN_RNN\data\beijing\traindata19851\Suitability1.tif"
UnifiedLineNumber(in_fn,criterion_fn,output_name)