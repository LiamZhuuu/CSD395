#!/bin/sh

import cv2
import os
import matplotlib.pyplot as plt
import subprocess

img_dir = '../images/'
img_path = img_dir + 'MD593_%04d_lossless_warped.tif' % (139)
img = cv2.imread(img_path)
plt.imshow(img)

height, width, channels = img.shape
border_width = 32
img_padded = cv2.copyMakeBorder(img, border_width, border_width, border_width, border_width, 
                                cv2.BORDER_CONSTANT, value=[0,0,0])
plt.imshow(img_padded)

clear = subprocess.Popen(['sh','clear_data.sh'])
clear.wait()

import caffe
import lmdb
import numpy as np

db_dir = '../data/'
db_train = lmdb.Environment(db_dir + 'md593_sample/md593_sample_train',map_size=int(1e12))
db_test = lmdb.Environment(db_dir + 'md593_sample/md593_sample_test',map_size=int(1e12))

txn_train = db_train.begin(write=True,buffers=True)
txn_test = db_test.begin(write=True,buffers=True)

count = 0
for x in range(32, 47):
    for y in range(32, 47):
        patch = img_padded[y-border_width : y+border_width+1 ,x-border_width:x+border_width+1]
        patch = patch[:, :, (2, 1, 0)]
        patch = patch.transpose((2, 0, 1))
        
        datum = caffe.proto.caffe_pb2.Datum()
        datum.height = border_width*2+1
        datum.width = border_width*2+1
        datum.channels = 3
        datum.data = patch.tostring()
        
        txn_train.put('%08d_%d_%d' % (count, x, y), datum.SerializeToString())
        if ((y-32) % 2 == 0) or ((y-32) % 2 == 0):
            txn_test.put('%08d_%d_%d' % (count, x, y), datum.SerializeToString())
        count = count + 1
        if count % 1000 == 0:
            # Write batch of images to database
            txn_train.commit()
            txn_train = db_train.begin(write=True)
            txn_test.commit()
            txn_test = db_test.begin(write=True)
            print 'Processed %i images.' % count

if count % 1000 != 0:
    # Write last batch of images
    txn_train.commit()
    txn_test.commit()
    print 'Processed a total of %i images.' % count
else:
    print 'Processed a total of %i images.' % count
    
db_train.close()
db_test.close()
