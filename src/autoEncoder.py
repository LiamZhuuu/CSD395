import caffe
import cv2
caffe.set_mode_gpu()

solver = caffe.SGDSolver('/home/jiaxuzhu/developer/CSD395/models/autoencoder/md593_autoencoder_solver.prototxt')
solver.solve()
print solver.net.blobs['deconv2'].data[0][0]
#print net.blobs['data'].data

'''
for i in range(5):
	solver.step(1)
	#print net.blobs['deconv1'].data
	print solver.net.blobs['loss'].data
'''

