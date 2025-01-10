# -*- coding:utf-8 -*-
# author: zyj time: 2021/7/12

from data_gen import *
from parameter import *
from data_process import d_preprocess
from normalization import noml

data_batch = np.zeros([batch_num, N, L], dtype='complex')
label_batch = np.zeros([batch_num, angle_num], dtype='float32')

count = 0
for delta_theta in range(100, 10000, 1):
    if delta_theta < 2000:
        num = 200
    elif delta_theta < 4000:
        num = 100
    elif delta_theta < 6000:
        num = 50
    elif delta_theta < 10000:
        num = 10
    label_batch[count:count+num, 0] = np.random.uniform(-60, 60 - (delta_theta / 100), (num))
    label_batch[count:count+num, 1] = label_batch[count:count+num, 0] + (delta_theta / 100)
    count += num
label_batch = np.sort(label_batch, axis=1)
np.random.shuffle(label_batch)

for i in range(len(train_snr)):
    for j in range(batch_num):
        data_batch[j] = data_gen_snr(label_batch[j][:], train_snr[i])
    label = label_batch
    data_path1 = 'data1_' + str(train_snr[i]) + '.npy'
    data_path2 = 'data2_' + str(train_snr[i]) + '.npy'
    label_path = 'label_' + str(train_snr[i]) + '.npy'
    data_batch = noml(data_batch)
    data = d_preprocess(data_batch)
    np.save(data_path1, data_batch)
    np.save(data_path2, data)
    np.save(label_path, label)