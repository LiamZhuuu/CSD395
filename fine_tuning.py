from texture.TextureClassifier import TextureClassifier

import logging

logging.basicConfig(level=logging.DEBUG)

data_dir = '/home/jiaxuzhu/data/landmark_patches'
p_labels = ['5N', '7n', '7N', '12N', 'Gr', 'LVe', 'Pn', 'SuVe', 'VLL']

tc = TextureClassifier(data_dir, p_labels, 1000)

model_dir = '/home/jiaxuzhu/developer/CSD395/model'
prefix = 'vgg16'
n_iter = 1

tc.mx_init(model_dir, prefix, n_iter)
tc.mx_training(0.0001, 8)