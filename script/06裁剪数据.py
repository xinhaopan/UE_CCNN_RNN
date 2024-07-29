from osgeo import gdal
import os

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

# 获取某目录下所有tif文件
def getTiffFileName(filepath, suffix):
    L1 = []
    L2 = []
    files = os.listdir(filepath)
    for file in files:  # 遍历刚获得的文件名files
        (filename, extension) = os.path.splitext(file)  # 将文件名拆分为文件名与后缀
        if (extension == suffix):  # 判断该后缀是否为.c文件
            L1.append(filepath + "/" + file)
            L2.append(filename)
    return L1, L2

simnames = ["urbanExpansion/plan1.tif","urbanExpansion/plan2.tif","urbanExpansion/plan3_w5.tif","urbanExpansion/cities/plan3.tif"]
cities = ["beijing","shijiazhuang","tianjin"]
i = 1
for simna in simnames:
    for city in cities:
        in_raster = "../" + simna
        out_raster = "../urbanExpansion/threecities/" + "plan" + str(i) + "_" + city + ".tif"
        mask_shp = "../origindata/"+ city +".shp"
        clip_raster(in_raster, out_raster, mask_shp)
        print(out_raster)
    i = i + 1