import caffe
caffe.set_mode_gpu()
solver = caffe.Net('/home/jiaxuzhu/developer/csd395/models/autoencoder/md593_autoencoder_solver.prototxt')
solver.solve()