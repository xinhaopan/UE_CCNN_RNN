import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import xarray as xr

weight = 6
plan = "plan3_w" + str(weight)
filepath = "../urbanExpansion/plan3_w" + str(weight)  + "_sp.tif"
arr = xr.open_rasterio(filepath).data[0,:,:]

arr[np.where(arr == 0)] = np.nan

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