from texture.TextureClassifier import TextureClassifier

import logging

logging.basicConfig(level=logging.DEBUG)

data_dir = '/home/jiaxuzhu/data/sample_patches'
p_labels = ['5N', '7n', '7N', '12N', 'Gr', 'LVe', 'Pn', 'SuVe', 'VLL']


model_dir = '/home/jiaxuzhu/developer/CSD395/model/inception-bn'
prefix = 'Inception_BN'
n_iter = 39

tc = TextureClassifier(data_dir, model_dir, prefix, n_iter, 32, 10)
tc.mx_training(50, 0.005, 32)
# tc.mx_predict('/home/jiaxuzhu/developer/CSD395/model', 'inception_texture', 3)