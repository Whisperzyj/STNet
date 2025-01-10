# -*- coding:utf-8 -*-
# author: zyj time: 2021/7/12

from parameter import *
import tensorflow as tf

def d_preprocess(data):
    data_num =data.shape[0]
    N = data.shape[1]
    L = data.shape[2]
    Rx = np.zeros([data_num, N, N], dtype='complex')
    ele_diag = np.zeros([data_num, N], dtype='float32')
    ele_real = np.zeros((data_num, round(N * (N - 1) / 2)), dtype='float32')
    ele_imag = np.zeros((data_num, round(N * (N - 1) / 2)), dtype='float32')
    ele = np.zeros((data_num, N * (N - 1)), dtype='float32')

    for i in range(data_num):
        r = np.mat(data[i])
        Rx[i] = (1 / L) * (r @ r.H)
        ele_diag[i] = np.diag(Rx[i]).real
        ele_r = []
        ele_i = []
        for m in range(1, N):
            for n in range(m):
                ele_r.append(Rx[i][m][n].real)
                ele_i.append(Rx[i][m][n].imag)
        ele_real[i] = ele_r
        ele_imag[i] = ele_i
    ele[:, ::2] = ele_real
    ele[:, 1::2] = ele_imag
    x = np.append(ele_diag, ele, axis=1)
    return (x)


