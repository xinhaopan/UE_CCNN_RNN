import arcpy
arcpy.CheckOutExtension("spatial")
arcpy.gp.overwriteOutput=1

mask= r"D:\Work\doing\CNN_RNN\data\beijing\origindata\boundary.shp"
filepath = r"D:\Work\doing\CNN_RNN\data\lulc\CLCD_v01_"
for i in range(1985,2018):
    raster = filepath + str(i) + ".tif"
    print(raster)
    out = r"D:\Work\doing\CNN_RNN\data\beijing\lulc\beijing" +  str(i) + ".tif"
    arcpy.gp.ExtractByMask_sa(raster, mask, out)
print("OK")