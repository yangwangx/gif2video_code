import os
import numpy as np

def rmse2psnr(rmse, maxVal=1.0):
    if rmse == 0:
        return 100
    else:
        return min(100, 20 * np.log10(maxVal/rmse))

def psnr2rmse(psnr, maxVal=1.0):
    if psnr >= 100:
        return 0
    else:
        return maxVal / 10 ** (psnr/20.0)
