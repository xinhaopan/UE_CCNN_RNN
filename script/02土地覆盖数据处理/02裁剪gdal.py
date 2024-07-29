# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 14:07:07 2022

@author: Xhpan
"""
from osgeo import gdal
def clip_raster(in_raster, out_raster, mask_shp):
    """
    :param in_raster: 输入栅格
    :param out_raster: 输出栅格
    :param mask_shp: 裁剪矢量
    :param wkid: wkid
    :return:
    """
    gdal.Warp(out_raster,
              in_raster,
              format='GTiff',
              dstSRS='EPSG:4326',
              cutlineDSName=mask_shp,
              cropToCutline=True,  # 按掩膜图层范围裁剪
              dstNodata=-9999,
              outputType=gdal.GDT_Float64)
    
filepath = r"D:\Work\doing\CNN_RNN\data\lulc\CLCD_v01_"
mask_shp =  r"D:\Work\doing\CNN_RNN\data\beijing\origindata\envelope.shp"
for i in range(1985,1986):
    in_raster = filepath +str(i) + ".tif"
    out_raster = r"D:\Work\doing\CNN_RNN\data\beijing\lulc\lulc" + str(i) + ".tif"
    clip_raster(in_raster, out_raster, mask_shp)
    print(out_raster)
