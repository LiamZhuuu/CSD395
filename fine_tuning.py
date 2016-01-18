from texture.TextureClassifier import TextureClassifier
import os
import logging

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

model_dir = '/home/jiaxuzhu/developer/CSD395/model/inception-bn'
prefix = 'inception-stage1'
n_iter = 6

tc = TextureClassifier(data_dir, model_dir, prefix, n_iter, 128, 10)
print tc.mx_confusion()