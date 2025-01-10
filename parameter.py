# -*- coding:utf-8 -*-
# author: zyj time: 2021/7/10

import numpy as np

N = 12  # 12 array elements
d_lamda = 0.5  # element spacing
L = 10   # number of snapshots
e_ratio = np.arange((N), dtype='complex')
train_snr = np.arange(0, 31, 5)
batch_num = 720000
data_num = batch_num*len(train_snr)

angle_num = 2
