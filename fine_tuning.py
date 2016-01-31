from texture.TextureClassifier import TextureClassifier
import os
import logging
import numpy as np
import mxnet as mx

logging.basicConfig(level=logging.DEBUG)

data_dir = '/home/jiaxuzhu/data/MD589_patches'
p_labels = ['5N', '7n', '7N', '12N', 'Gr', 'LVe', 'Pn', 'SuVe', 'VLL', 'BackG']


# model_dir = '/home/jiaxuzhu/models/Inception_BN'
# prefix = 'Inception_BN'
# n_iter = 39

# tc = TextureClassifier(model_dir, prefix, n_iter, 10)
# tc.mx_init(data_dir, 128)
# ctx = [mx.gpu(i) for i in range(1)]
# tc.mx_training(10, 0.001, 128, os.path.join(model_dir, 'inception-stage1'), ctx)

# tc.mx_predict('/home/jiaxuzhu/developer/CSD395/model', 'inception_texture', 3)

model_dir = '/home/jiaxuzhu/develope/CSD395/model_publish'
prefix = 'inception-stage1'
n_iter = 10
#

tc = TextureClassifier(model_dir, prefix, n_iter, 10)
tc.mx_init(data_dir, 128)
print  tc.mx_confusion('/home/jiaxuzhu/data/MD589_patches/test.lst')
# # print np.argmax(tc.mx_predict('/home/jiaxuzhu/data/sample_patches/test.rec', 128), axis=1)
# data = np.load('/home/jiaxuzhu/data/landmark_patches/5N_patches.npy')
# data = data.transpose(0, 3, 1, 2)
# print np.argmax(tc.mx_predict_np(data, 64), axis=1)
