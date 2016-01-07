import mxnet
import numpy as np
import os

data_dir = '/home/jiaxuzhu/data/landmark_patches'
pathes = np.load(os.path.join(data_dir, '5N_patches.npy'))
print pathes.shape