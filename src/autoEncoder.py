import caffe
import cv2
caffe.set_mode_gpu()

solver = caffe.SGDSolver('/home/jiaxuzhu/developer/CSD395/models/autoencoder/md593_autoencoder_solver.prototxt')
solver.solve()


filters = solver.net.params['conv1'][0].data
print filters
img = solver.net.blobs['deconv1'].data[9].reshape(100,100) * 255
cv2.imwrite('recon.jpg', img)
#print net.blobs['data'].data

'''
for i in range(5):
	solver.step(1)
	#print net.blobs['deconv1'].data
	print solver.net.blobs['loss'].data
'''

