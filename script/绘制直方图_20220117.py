# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 21:44:34 2021

@author: Xhpan
"""

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import xarray as xr


plans = ["plan1","plan2","plan3"]
for plan in plans:
    if plan == "plan1":
        filepath = "../Unet_result/Suitability_1_2015_lb.tif"
    elif plan == "plan2":
        filepath = "../LSTMdata/LSTM2015_nor.tif"
    elif plan == "plan3":
        filepath = "../Unet_result/suitability_plan3.tif"

    arr = xr.open_rasterio(filepath).data[0,:,:]

    arr[np.where(arr == 0)] = np.nan
    print(plan + str(np.sum(arr > 0)))
    # ----------------------------------------------------------------------------------
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

    #---------------------------------------------------------------------------------------
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