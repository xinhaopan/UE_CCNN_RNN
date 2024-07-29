import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib
from osgeo import gdal, osr
from osgeo import gdal_array
import pandas as pd


# 将读取文件的灰度矩阵，转化为直方图，这里的直方图定义为python的dict类型，索引为灰度级，值为对应的概率。
def arrayToHist(grayArray, nums):
    if (len(grayArray.shape) != 2):
        print("length error")
        return None
    rows, cols = grayArray.shape
    hist = {}
    for k in range(nums):
        hist[k] = 0
    for i in range(rows):
        for j in range(cols):
            if (hist.get(grayArray[i][j]) is None):
                hist[grayArray[i][j]] = 0
            hist[grayArray[i][j]] += 1
    # normalize
    n = rows * cols
    for key in hist.keys():
        try:
            hist[key] = float(hist[key]) / n
        except:
            pass
    return hist


# 传入的直方图要求是个字典，每个灰度对应着概率
def drawHist(hist, name):
    keys = hist.keys()
    values = hist.values()
    x_size = len(hist) - 1  # x轴长度，也就是灰度级别
    '''
    axis_params = []
    axis_params.append(0)
    axis_params.append(x_size)
    '''
    # plt.figure()
    if name != None:
        plt.title(name)
    plt.bar(tuple(keys), tuple(values))  # 绘制直方图
    # plt.show()


# 传入图片数组进行imshow
def drawImage(image, title):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')
    plt.title(title)


def readTifAsArray(tifPath):
    dataset = gdal.Open(tifPath)
    if dataset == None:
        print(tifPath + "文件错误")
        return tifPath

    image_datatype = dataset.GetRasterBand(1).DataType
    row = dataset.RasterYSize
    col = dataset.RasterXSize
    nb = dataset.RasterCount
    proj = dataset.GetProjection()
    gt = dataset.GetGeoTransform()

    if nb != 1:
        array = np.zeros((row, col, nb),
                         dtype=gdal_array.GDALTypeCodeToNumericTypeCode(
                             image_datatype))
        for b in range(nb):
            band = dataset.GetRasterBand(b + 1)
            nan = band.GetNoDataValue()
            array[:, :, b] = band.ReadAsArray()
    else:
        array = np.zeros((row, col),
                         dtype=gdal_array.GDALTypeCodeToNumericTypeCode(
                             image_datatype))
        band = dataset.GetRasterBand(1)
        nan = band.GetNoDataValue()
        array = band.ReadAsArray()
    return array, nan, gt, proj


#  写入tif函数
def array2raster(TifName, GeoTransform, array):
    cols = array.shape[1]  # 矩阵列数
    rows = array.shape[0]  # 矩阵行数
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(TifName, cols, rows, 1, gdal.GDT_Float32)
    # 括号中两个0表示起始像元的行列号从(0,0)开始
    outRaster.SetGeoTransform(tuple(GeoTransform))
    # 获取数据集第一个波段，是从1开始，不是从0开始
    outRaster.GetRasterBand(1).SetNoDataValue(-32768)
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    # 代码4326表示WGS84坐标
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


# 直方图匹配函数，接受原始图像数组和目标灰度直方图
def histMatch(grayOriginal, histTarget):
    # 计算目标图像累计直方图
    tmp = 0.0
    histTarget_acc = histTarget.copy()
    grayTarget_max = max(histTarget)
    for i in range(grayTarget_max + 1):
        tmp += histTarget[i]
        histTarget_acc[i] = tmp
    # 计算原始影像的累计直方图
    grayOriginal = grayOriginal - grayOriginal.min()
    grayOriginal_max = grayOriginal.max()
    histOriginal = arrayToHist(grayOriginal, grayOriginal_max)
    tmp = 0.0
    histOriginal_acc = histOriginal.copy()
    for i in range(grayOriginal_max + 1):
        tmp += histOriginal[i]
        histOriginal_acc[i] = tmp
    # 计算映射
    if grayTarget_max > grayOriginal_max:
        grayTarget_max = grayTarget_max
    else:
        grayTarget_max = grayOriginal_max
    M = np.zeros(grayTarget_max + 1)
    for i in range(grayTarget_max + 1):
        idx = 0
        minv = 1
        for j in histTarget_acc:
            if (np.fabs(histTarget_acc[j] - histOriginal_acc[i]) < minv):
                minv = np.fabs(histTarget_acc[j] - histOriginal_acc[i])
                idx = int(j)
        M[i] = idx
    des = M[grayOriginal]
    return des


# 直方图匹配函数，接受原始图像数组和目标灰度直方图,适用于MERSI-2数据
def histMatchMERSI2(grayOriginal, histTarget):
    # 计算目标图像累计直方图
    tmp = 0.0
    histTarget_acc = histTarget.copy()
    grayTarget_max = max(histTarget)
    for i in range(grayTarget_max + 1):
        tmp += histTarget[i]
        histTarget_acc[i] = tmp
    # 计算原始影像的累计直方图
    # grayOriginal = grayOriginal - grayOriginal.min()
    # grayOriginal_max = grayOriginal.max()
    histOriginal = arrayToHist(grayOriginal, grayTarget_max + 1)
    tmp = 0.0
    histOriginal_acc = histOriginal.copy()
    for i in range(grayTarget_max + 1):
        tmp += histOriginal[i]
        histOriginal_acc[i] = tmp
    # 计算映射
    M = np.zeros(grayTarget_max + 1)
    for i in range(grayTarget_max + 1):
        idx = 0
        minv = 1
        for j in histTarget_acc:
            if (np.fabs(histTarget_acc[j] - histOriginal_acc[i]) < minv):
                minv = np.fabs(histTarget_acc[j] - histOriginal_acc[i])
                idx = int(j)
        M[i] = idx
    des = M[grayOriginal]
    return des


# 这部分写的很乱主要是为了测试一下- -
if __name__ == '__main__':
    imdir = "../LSTMdata/LSTM2015_nor.tif"
    image = readTifAsArray(imdir)[0][2000:2100, 2000:2100]
    image = image - image.min()
    imageMax = image.max()
    hist = arrayToHist(image, imageMax)


    def dictToList_drawHist(hist):  # hist 为字典类型
        List = []
        for k in hist:
            List.append(hist[k])
        df = pd.DataFrame(List, columns=['value'])
        # print('histogram_sunm:', df.sum()) # 测试直方图累计频率是否为1
        drawHist(hist, 'hist')


    '''
    dictToList_drawHist(hist)
    dictToList_drawHist(histTarget)
    '''
    imageTarget = readTifAsArray(imdir)[0][3000:3100, 3000:3100]
    imageTarget = imageTarget - imageTarget.min()
    imageTargetMax = imageTarget.max()
    histTarget = arrayToHist(imageTarget, imageTargetMax)

    # 开始绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 8))
    # 原始图和直方图
    plt.subplot(2, 3, 1)
    plt.title("原始图片")
    plt.imshow(image, cmap='gray')

    plt.subplot(2, 3, 4)
    drawHist(hist, "原始直方图")

    # match图和其直方图
    plt.subplot(2, 3, 2)
    plt.title("match图片")
    plt.imshow(imageTarget, cmap='gray')

    plt.subplot(2, 3, 5)
    drawHist(histTarget, "match直方图")

    # match后的图片及其直方图
    im_d = histMatch(image, histTarget)  # 将目标图的直方图用于给原图做均衡，也就实现了match
    plt.subplot(2, 3, 3)
    plt.title("match后的图片")
    plt.imshow(im_d, cmap='gray')

    plt.subplot(2, 3, 6)
    im_d = im_d.astype(np.int16)
    hist_d = arrayToHist(im_d, im_d.max())
    drawHist(hist_d, "match后的直方图")

    plt.show()
