import mxnet as mx
import numpy as np
import os
import shutil

class FeatureExtractor:
    def __init__(self, model_dir, batch_size=1, ctx='cpu'):
        model = mx.model.FeedForward.load(model_dir, 1)
        internals = model.symbol.get_internals()
        fea_symbol = internals["fc7_output"]
        if ctx == 'cpu':
            ctx = mx.cpu()
        else:
            ctx = mx.gpu()
        self.net = mx.model.FeedForward(ctx=ctx, symbol=fea_symbol, numpy_batch_size=batch_size,
                                        arg_params=model.arg_params, aux_params=model.aux_params, allow_extra_params=True)

    def extract(self, images):
        return self.net.predict(images)


if __name__ == '__main__':
    data_dir = '/home/jiaxuzhu/data/landmark_patches'
    model_dir = '/home/jiaxuzhu/developer/CSD395/model/vgg16'
    # names = ['5N', '7n', '7N', '12N', 'Gr', 'LVe', 'Pn', 'SuVe', 'VLL']

    fe = FeatureExtractor(model_dir, batch_size=16, ctx='gpu')
    for root, dirs, files in os.walk(data_dir):
        break
    dst = os.path.join(data_dir, 'features')
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.mkdir(dst)
    for name in files:
        type = name.split('.')[-1]
        prefix = name.split('.')[0]
        if type != 'npy':
            continue
'''
    data_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_patches'
    model_dir = '/oasis/projects/nsf/csd395/jiaxuzhu/models/vgg16/vgg16'
    for root, dirs, files in os.walk(data_dir):
        break

    fe = FeatureExtractor(model_dir, batch_size=16, ctx='cpu')

    for name in files:
        types = name.split('.')[-1]
        if types != 'npy':
            continue
        prefix = name.split('.')[0]
'''
        images = np.load(os.path.join(data_dir, name)).transpose(0, 3, 1, 2)
        print images.shape

        features = fe.extract(images)
        print features.shape

        np.save(os.path.join(dst, '%s_features.npy' % prefix), features)
'''        np.save(os.path.join(data_dir, '%s_features.npy' % prefix), features)
'''