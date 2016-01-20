from texture.TextureClassifier import TextureClassifier
import os
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)

data_dir = '/home/jiaxuzhu/data/sample_patches'
p_labels = ['5N', '7n', '7N', '12N', 'Gr', 'LVe', 'Pn', 'SuVe', 'VLL']


# model_dir = '/home/jiaxuzhu/developer/CSD395/model/inception-bn'
# prefix = 'Inception_BN'
# n_iter = 39
#
# tc = TextureClassifier(data_dir, model_dir, prefix, n_iter, 128, 10)
# tc.mx_training(10, 0.001, 128, os.path.join(model_dir, 'inception-stage1'))
#
# tc.mx_predict('/home/jiaxuzhu/developer/CSD395/model', 'inception_texture', 3)

model_dir = '/home/jiaxuzhu/developer/CSD395/model_publish'
prefix = 'inception-stage1'
n_iter = 6

tc = TextureClassifier(model_dir, prefix, n_iter, 10)
tc.mx_init(data_dir, 128)
# tc.mx_confusion()
print np.argmax(tc.mx_predict('/home/jiaxuzhu/data/sample_patches/test.rec', 128), axis=1)