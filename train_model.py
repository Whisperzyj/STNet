# -*- coding:utf-8 -*-
# author: zyj time: 2021/7/12

import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os
from scipy import io
from parameter import *

from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras.models import Model
from keras import layers, Input, optimizers, callbacks, losses

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
IDS = [0]

r_label = []
r_data1 = []
r_data2 = []
for i in range(len(train_snr)):
    file_data1 = 'data1_' + str(train_snr[i]) + '.npy'
    file_data2 = 'data2_' + str(train_snr[i]) + '.npy'
    file_label = 'label_' + str(train_snr[i]) + '.npy'
    r_data1.append(np.load(file_data1))
    r_data2.append(np.load(file_data2))
    r_label.append(np.load(file_label))

r_data1 = np.reshape(r_data1, (data_num, N, L))
data1 = np.transpose(r_data1, (0, 2, 1))
data1_real = np.zeros([data_num, 2*L, N], dtype='float32')
data1_real[:, ::2, :] = data1.real
data1_real[:, 1::2, :] = data1.imag
data2 = np.reshape(r_data2, (data_num, N**2))
label = np.reshape(r_label, (data_num, angle_num))

tbcallbacks =[
    callbacks.TensorBoard(
        log_dir='my_log_dir',
        histogram_freq=1,
        embeddings_freq=1,
        write_grads=True,
    )
]

input1 = data1_real
input2 = data2
output = label

input_re1 = Input(shape=(2*L, N), dtype='float32', name='receive1')
input_re2 = Input(shape=(N*N), dtype='float32', name='receive2')

x1 = layers.LSTM(32, return_sequences=True)(input_re1)
x1 = layers.LSTM(64, return_sequences=True)(x1)
x1 = layers.Dropout(0.5)(x1)
x1 = layers.LSTM(64, return_sequences=True)(x1)
x1 = layers.Dropout(0.5)(x1)
x1 = layers.LSTM(32)(x1)

x_shortcut1 = input_re2

x2 = layers.Dense(256, activation='relu')(input_re2)
x2 = layers.Dense(128, activation='relu')(x2)

x3 = layers.concatenate([x2, x_shortcut1])
x_shortcut2 = x3

x3 = layers.Dense(256, activation='relu')(x3)
x3 = layers.Dense(128, activation='relu')(x3)

x4 = layers.concatenate([x3, x_shortcut2])

x4 = layers.Dense(256, activation='relu')(x4)
x4 = layers.Dense(128, activation='relu')(x4)

x5 = layers.concatenate([x1, x4])

x5 = layers.Dense(256, activation='relu')(x5)
x5 = layers.Dense(128, activation='relu')(x5)
x5 = layers.Dense(32, activation='relu')(x5)
x5 = layers.Dense(8, activation='relu')(x5)

angle_pre = layers.Dense(2, name='angle')(x5)

def main():
    model = Model([input_re1, input_re2], angle_pre)

    def scheduler(epoch):
        # 每隔100个epoch，学习率减小为原来的1/10
        if epoch % 10 == 0 and epoch != 0:
            lr = keras.backend.get_value(model.optimizer.lr)
            keras.backend.set_value(model.optimizer.lr, lr * 0.7)
            print("lr changed to {}".format(lr * 0.7))
        return keras.backend.get_value(model.optimizer.lr)

    model.summary()
    learning_rate_reduction = LearningRateScheduler(scheduler)
    # learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1, factor=0.7)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=['mse'], metrics=['acc'])
    history = model.fit([input1, input2], output, epochs=100, batch_size=256, verbose=1, validation_split=0.1, callbacks=[tbcallbacks, learning_rate_reduction])
    model.save('DNN.h5')
    plt.figure(1)
    plt.plot(np.array(history.history['loss']), label='Train')
    plt.plot(np.array(history.history['val_loss']), label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()
    plt.figure(2)
    plt.plot(np.array(history.history['acc']), label='Train')
    plt.plot(np.array(history.history['val_acc']), label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()
    io.savemat('loss_acc.mat', {'loss': history.history['loss'], 'val_loss': history.history['val_loss'],
                                'acc': history.history['acc'], 'val_acc': history.history['val_acc']})

if __name__ == '__main__':
    main()
