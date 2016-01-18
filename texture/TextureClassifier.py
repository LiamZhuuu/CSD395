import sys
sys.path.insert(0, '/home/jiaxuzhu/developer/mxnet/python')
import os
import numpy as np
import scipy
import mxnet as mx
import cv2


def data_separation(data):
    n_data = len(data)
    train_data = data[:n_data/2, :]
    val_data = data[n_data/2:n_data/4*3, :]
    test_data = data[n_data/4*3:, :]
    return {'train': train_data, 'val': val_data, 'test': test_data}


class TextureClassifier:
    def __init__(self, data_dir, model_dir, prefix, n_iter, b_size, n_class):
        sets = ['train', 'eval', 'test']
        self.iter = {}
        for dataset in sets:
            rec_file = os.path.join(data_dir, '%s.rec' % dataset)
            self.iter[dataset] = mx.io.ImageRecordIter(
                path_imgrec=rec_file,
                batch_size=b_size,
                shuffle=True,
                data_shape=(3, 224, 224)
            )
        self.init_model = mx.model.FeedForward.load(os.path.join(model_dir, prefix), n_iter)
        internals = self.init_model.symbol.get_internals()
        symbol = internals['flatten_output']
        symbol = mx.symbol.FullyConnected(data=symbol, name='fullc', num_hidden=n_class)
        self.symbol = mx.symbol.SoftmaxOutput(data=symbol, name='softmax')
        self.net = None

    def mx_training(self, n_epoch, l_rate, b_size):
        opt = mx.optimizer.SGD(learning_rate=l_rate)
        self.net = mx.model.FeedForward(ctx=mx.gpu(), symbol=self.symbol, num_epoch=n_epoch, optimizer=opt,
                                        arg_params=self.init_model.arg_params, aux_params=self.init_model.aux_params,
                                        allow_extra_params=True)
        self.net.fit(self.iter['train'], eval_data=self.iter['eval'],
                     batch_end_callback=mx.callback.Speedometer(b_size, 30))

        labels = self.net.predict(self.iter['train'])
        print self.net.arg_params['fullc_weight']
        print labels[:10, :]
        labels = np.argmax(labels, axis=1)
        print labels

        self.net.save('/home/jiaxuzhu/developer/CSD395/model/inception_texture')
