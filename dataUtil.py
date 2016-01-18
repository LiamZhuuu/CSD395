import numpy as np
import os
import shutil
import cv2

def data_sampling(data_dir, dst_dir, p_labels, n_sample):

    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.mkdir(dst_dir)

    files = {}
    files['train'] = open(os.path.join(dst_dir, 'train_list.txt'), 'w')
    files['test'] = open(os.path.join(dst_dir, 'test_list.txt'), 'w')
    files['eval'] = open(os.path.join(dst_dir, 'eval_list.txt'), 'w')

    idx = 0
    lists = {'train': [], 'test': [], 'eval': []}
    for j in range(len(p_labels)):
            p_name = p_labels[j]
            print p_name
            sub_dst = os.path.join(dst_dir, p_name)
            if os.path.exists(sub_dst):
                shutil.rmtree(sub_dst)
            os.mkdir(sub_dst)

            patches = np.load(os.path.join(data_dir, '%s_patches.npy' % p_name))
            print patches.shape
            # patches = np.transpose(patches, (0, 3, 1, 2))
            shuffle = np.random.permutation(len(patches))
            patches = patches[shuffle[:n_sample], :]

            for i in range(len(patches)):
                image_path = os.path.join(sub_dst, '%s_%08d.jpg' % (p_name, i))
                cv2.imwrite(image_path, patches[i, :])
                if i < (n_sample / 2):
                    lists['train'].append('%d\t%d\t%s\n' % (idx, j, '%s/%s_%08d.jpg' % (p_name, p_name, i)))
                elif i < (n_sample / 4 * 3):
                    lists['eval'].append('%d\t%d\t%s\n' % (idx, j, '%s/%s_%08d.jpg' % (p_name, p_name, i)))
                else:
                    lists['test'].append('%d\t%d\t%s\n' % (idx, j, '%s/%s_%08d.jpg' % (p_name, p_name, i)))
                idx += 1

    for key,value in lists.items():
        idx = np.random.permutation(len(value))
        for i in range(len(idx)):
            files[key].write(value[idx[i]])
        files[key].close()

# def data_loading(image_dir, p_labels, n_sample, ):



data_dir = '/home/jiaxuzhu/data/landmark_patches'
dst_dir = '/home/jiaxuzhu/data/sample_patches'
p_labels = ['5N', '7n', '7N', '12N', 'Gr', 'LVe', 'Pn', 'SuVe', 'VLL', 'BackG']

n_sample = 1024
data_sampling(data_dir, dst_dir, p_labels, n_sample)

# for root, dirs, files in os.walk(data_dir):
#     break
#
# bg = None
# print files
# count = 0
# for file in files:
#     prefix = file.split('_')[0]
#     # print prefix
#     if prefix == 'patchesBg':
#         tmp = np.load(os.path.join(data_dir, file))
#         idx = np.random.permutation(len(tmp))[:10]
#         tmp = tmp[idx, :]
#         if bg is None:
#             bg = tmp
#         else:
#             bg = np.vstack((bg, tmp))
#         count += 1
#         print count
# np.save(os.path.join(data_dir, 'BackG_patches.npy'), bg)
