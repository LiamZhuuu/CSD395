import numpy as np
import os

data_dir = '/home/jiaxuzhu/data/landmark_patches'
names = {'5N': 0, '7n': 1, '7N': 2,
         '12N': 3, 'Gr': 4, 'LVe': 6,
         'Pn': 7, 'SuVe': 8, 'VLL': 9
         }
for root, dirs, files in os.walk(data_dir):
    break

features = {}
for data in files:
    data_type = data.split('.')[-1]
    if data_type != 'npy':
        continue
    data_prefix = data.split('_')[1]
    feature = np.load(os.path.join(data_dir, data))
    if data_prefix not in features:
        features[data_prefix] = feature
    else:
        features[data_prefix] = np.vstack((features[data_prefix], feature))

for (key, value) in features.items():
    print key, value.shape
    patch_file = '%s_patches.npy' % key
    np.save(os.path.join(data_dir, patch_file), value)

