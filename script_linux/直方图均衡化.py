from skimage import exposure
# Equalization
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

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
    data = np.stack(L, 0)
    if nb == 1:
        data = data[0, :, :]

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

filepath = "../LSTMdata/LSTM2015_nor.tif"
arr = LoadData(filepath)
arr_boundary = LoadData("../inputdata/boundary/boundary.tif")

# arr = exposure.equalize_hist(arr)
# arr = exposure.equalize_adapthist(arr, clip_limit=0)
p2, p98 = np.percentile(arr, (2, 99.997))
arr = exposure.rescale_intensity(arr, in_range=(p2, p98))

resultpath = "../LSTMdata/LSTM2015_nor1.tif"

arr_boundary = LoadData("../inputdata/boundary/boundary.tif")
arr[np.where(arr_boundary == 0)] = 0

WriteTiff(arr,filepath, resultpath)
plan = "plan4"

num = []
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if arr[i][j] >= 0:
            num.append(arr[i][j])


font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)

group = []
for n in range(0,101):
    group.append(n/100)



plt.hist(num, group, histtype='bar', rwidth=0.8)

# plt.legend()

plt.xlabel('')
plt.ylabel('')


plt.title(plan + '_全部', FontProperties=font)
plt.savefig("../urbanExpansion/" + plan + '_全部'+ ".png")
plt.show()

num = []
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if arr[i][j] > 0.01 and arr[i][j] < 0.99:
            num.append(arr[i][j])


font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)

group = []
for n in range(0,101):
    group.append(n/100)

plt.hist(num, group, histtype='bar', rwidth=0.8)

# plt.legend()

plt.xlabel('')
plt.ylabel('')

plt.title(plan + '_0.01-0.99', FontProperties=font)
plt.savefig("../urbanExpansion/" + plan + '_0.01-0.99' + ".png")
plt.show()

