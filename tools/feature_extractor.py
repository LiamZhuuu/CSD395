import mxnet as mx
import numpy as np
import os
data_dir = '/home/jiaxuzhu/data/landmark_patches'
model_dir = '/home/jiaxuzhu/developer/CSD395/model'
names = ['5N', '7n', '7N', '12N', 'Gr', 'LVe', 'Pn', 'SuVe', 'VLL']

model = mx.model.FeedForward.load(model_dir + '/vgg16', 1, ctx=mx.gpu())
internals = model.symbol.get_internals()
fea_symbol = internals["fc7_output"]
feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, numpy_batch_size=16, 
	arg_params=model.arg_params, aux_params=model.aux_params, allow_extra_params=True)

for name in names:
	images = np.load(os.path.join(data_dir, '%s_patches.npy' % name)).transpose(0,3,1,2)
	print images.shape
	features = feature_extractor.predict(images)
	print features.shape
	np.save(os.path.join(data_dir, '%s_features.npy' % name), features)