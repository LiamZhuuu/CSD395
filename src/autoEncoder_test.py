import caffe
import cv2
caffe.set_mode_gpu()
net = caffe.Net('../models/autoencoder/md593_autoencoder_test.prototxt','../models/autoencoder/md593_autoencoder_iter_1000.caffemodel', caffe.TEST)

img = cv2.imread('../images/md593_test.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

net.blobs['data'].data[...] = img * 0.00392156862
print net.blobs['data'].data

out = net.forward()

tmp = out['deconv2'].reshape(225,225)
print tmp.max()*255, tmp.min()*255

cv2.imwrite('recon.jpg', tmp * 255)


