# -*- coding:utf-8 -*-
# author: zyj time: 2022/4/3

import numpy as np

def noml(data):
    [x, y, z] = data.shape
    data_real = data.real
    data_imag = data.imag
    for i in range(x):
        for j in range(y):
            data_real[i, j, :] = data_real[i, j, :] - np.mean(data_real[i, j, :])
            data_real[i, j, :] = data_real[i, j, :] / np.max(data_real[i, j, :])
            data_imag[i, j, :] = data_imag[i, j, :] - np.mean(data_imag[i, j, :])
            data_imag[i, j, :] = data_imag[i, j, :] / np.max(data_imag[i, j, :])
    data_normal = data.real + 1j*data.imag
    return (data_normal)