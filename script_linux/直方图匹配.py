import histogram_matching
import numpy as np

#影像共8000行,40探元并扫,每个扫描带包括200个影像行
imageDir = "../LSTMdata/LSTM2015_nor.tif"
imageOriginal = histogram_matching.readTifAsArray(imageDir)
minPercent5 = np.percentile(imageOriginal[0], 5)
maxPercent95 = np.percentile(imageOriginal[0], 95) #取置信区间为95%的值作为最大值
imageOriginal = np.where((imageOriginal[0] >= minPercent5) & (imageOriginal[0] <= maxPercent95),
                         imageOriginal[0], imageOriginal[0].mean())
imageOriginal = imageOriginal - imageOriginal.min() #将灰度最小值转换为0
rows, cols = imageOriginal.shape
imageTarget = imageOriginal[0:200,:].astype(np.int16)
histTarget = histogram_matching.arrayToHist(imageTarget, maxPercent95.astype(np.int16))

subImagesList = np.arange(1,40)
for subImage in subImagesList:
    startRow = subImage * 200
    overRow = startRow + 200
    print(startRow, overRow)
    imageOriginal[startRow:overRow, :] = histogram_matching.histMatchMERSI2(
                                        imageOriginal[startRow:overRow, :].astype(np.int16),
                                        histTarget)
histMatchImage = imageOriginal + minPercent5
histogram_matching.array2raster("../LSTMdata/LSTM2015_nor1.tif", [0, 1, 0, 0, 0, 1], histMatchImage)
