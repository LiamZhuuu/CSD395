import numpy as np
import os
import shutil
import cv2
import random


def write_image(data_dir, dst_dir, p_labels):
    for root, dirs, files in os.walk(data_dir):
        break
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.mkdir(dst_dir)

    train_list = []
    val_list = []
    test_list = []

    total = 0
    label_count = {}
    for label in p_labels:
        label_count[label] = 0
    label_idx = {}
    label_list = {}
    for i in range(len(p_labels)):
        label_idx[p_labels[i]] = i
        label_list[p_labels[i]] = []

    for data in files:
        print data
        prefix = data.split('.')[0].split('_')
        if prefix[0] == 'patches':
            label = prefix[1]
        else:
            label = 'BackG'
        section = int(prefix[2])
        images = np.load(os.path.join(data_dir, data))
        label_dir = os.path.join(dst_dir, label)
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)
        count = 0
        for image in images:
            tmp = image.transpose(2, 0, 1)
            image = np.vstack((tmp[2:3, :], tmp[1:2, :], tmp[0:1, :]))
            image = image.transpose(1, 2, 0)
            image_name = '%s_%d_%08d.jpg' % (label, section, count)
            cv2.imwrite(os.path.join(label_dir, image_name), image)
            list_line = '%d\t%d\t%s\n' % (total, label_idx[label], os.path.join(label, image_name))
            label_list[label].append(list_line)
            label_count[label] += 1
            count += 1
            total += 1

    for label in p_labels:
        random.shuffle(label_list[label])
        if label == 'BackG':
            train_list.extend(label_list[label][:10240])
            val_list.extend(label_list[label][10240:11520])
            test_list.extend(label_list[label][11520:])
        else:
            num = len(label_list[label])
            train_list.extend(label_list[label][:int(num*0.7)])
            val_list.extend(label_list[label][int(num*0.7):int(num*0.8)])
            test_list.extend(label_list[label][int(num*0.8):])

    random.shuffle(train_list)
    random.shuffle(val_list)
    random.shuffle(test_list)
    with open(os.path.join(dst_dir, 'train.lst'), 'w') as train_file:
        train_file.writelines(train_list)
    with open(os.path.join(dst_dir, 'val.lst'), 'w') as val_file:
        val_file.writelines(val_list)
    with open(os.path.join(dst_dir, 'test.lst'), 'w') as test_file:
        test_file.writelines(test_list)
    print label_count
# def data_loading(image_dir, p_labels, n_sample, ):

def gen_lst(data_dir, p_labels):
    train_list = []
    val_list = []
    test_list = []
    idx = 0
    for i in range(len(p_labels)):
        dir = p_labels[i]
        for root, subdirs, files in os.walk(os.path.join(data_dir, dir)):
    	    break
	n_file = len(files)
	print dir, len(files)
	random.shuffle(files)
        for j in range(len(files)):
	    image = files[j]
	    if image.split('.')[-1] != 'jpg':
		continue
	    line = '%d\t%d\t%s\n' % (idx, i, dir + '/' + image)
	    if j < int(n_file*0.7):
                train_list.append(line)
	    elif j < int(n_file*0.8):
		val_list.append(line)
	    else:
		test_list.append(line)
            idx += 1
        
        dir = p_labels[i] + '_surround'
        for root, subdirs, files in os.walk(os.path.join(data_dir, dir)):
            break
        n_file = len(files)
        print dir, len(files)
        random.shuffle(files)
        for j in range(len(files)):
            image = files[j]
            if image.split('.')[-1] != 'jpg':
                continue
            line = '%d\t%d\t%s\n' % (idx, 9, dir + '/' + image)
            if j < int(n_file*0.7):
                train_list.append(line)   
            elif j < int(n_file*0.8):
                val_list.append(line)
            else:
                test_list.append(line)
	    idx += 1

    random.shuffle(train_list)
    random.shuffle(val_list)
    random.shuffle(test_list)
    with open(os.path.join(data_dir, 'train.lst'), 'w') as train_file:
        train_file.writelines(train_list)
    with open(os.path.join(data_dir, 'val.lst'), 'w') as val_file:
        val_file.writelines(val_list)
    with open(os.path.join(data_dir, 'test.lst'), 'w') as test_file:
        test_file.writelines(test_list)


	    
                


data_dir = '/home/jiaxuzhu/data/MD589_patches'
dst_dir = '/home/jiaxuzhu/data/all_patches'
p_labels = ['5N', '7n', '7N', '12N', 'Gr', 'LVe', 'Pn', 'SuVe', 'VLL']

n_sample = 1024
gen_lst(data_dir, p_labels)
# data_sampling(data_dir, dst_dir, p_labels, n_sample)
# write_image(data_dir, dst_dir, p_labels)
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
