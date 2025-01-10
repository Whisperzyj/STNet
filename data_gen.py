# -*- coding:utf-8 -*-
# author: zyj time: 2021/7/10

import numpy as np
import scipy
from parameter import *

def data_gen_comatrix(theta, snr):
    s_n = 10**(snr/20)
    S_cov = np.repeat([1], angle_num)
    S = np.diag(S_cov)
    A = np.zeros((N, angle_num), dtype='complex')

    for i in range(angle_num):
        A[:, i] = np.exp(2j * np.pi * 0.5 * np.sin(theta[i] * np.pi / 180) * e_ratio)

    N_cov = np.repeat([1/s_n], N)
    n = np.diag(N_cov)

    A = np.mat(A)

    X = A@S@A.H + n
    return (X)

def data_gen_snr(theta, snr):
    s_n = 10**(snr/20)

    A = np.zeros((N, angle_num), dtype='complex')
    S = np.zeros((angle_num, L), dtype='complex')
    n = np.zeros((N, L), dtype='complex')
    for i in range(angle_num):
        A[:, i] = np.exp(2j * np.pi * 0.5 * np.sin(theta[i] * np.pi / 180) * e_ratio)
        S[i, :] = np.random.normal(0, 1, (1, L)) + 1j*np.random.normal(0, 1, (1, L))
    for i in range(N):
        n[i, :] = (1/s_n)*(np.random.normal(0, 1, (1, L)) + 1j*np.random.normal(0, 1, (1, L)))
    X = np.dot(A, S) + n

    return (X)