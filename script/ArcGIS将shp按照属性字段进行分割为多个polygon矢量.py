# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 16:37:16 2022

@author: Xhpan
"""
    
from osgeo import ogr
import os
shpfile = r"D:\Work\otherWork\wyh\光污染\青藏高原重点关注自然区划\数据\国家级自然保护区\NNR479_TP.shp"  
resultpath = r"D:\Work\otherWork\wyh\boundary\natureReserve"
if not os.path.exists(resultpath):
	os.makedirs(resultpath)

driver = ogr.GetDriverByName("ESRI Shapefile")
ds = ogr.Open(shpfile)
layer = ds.GetLayer(0)
for i in range(layer.GetFeatureCount()):
    source_feats = layer.GetFeature(i)
    source_id = source_feats.GetField('id2') # 获取每个面cyid字段值
    layer.SetAttributeFilter("id2 = {}".format(source_id))
    
    extfile = resultpath + "/" + str(source_id).zfill(2) + ".shp"
    newds = driver.CreateDataSource(extfile)
    lyrn = newds.CreateLayer('rect', None, ogr.wkbPolygon)
    
    feat = layer.GetNextFeature()
    while feat is not None:
        lyrn.CreateFeature(feat)
        feat = layer.GetNextFeature()
    newds.Destroy()
    print(i)
        




