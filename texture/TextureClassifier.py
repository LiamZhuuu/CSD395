import sys
sys.path.insert(0, '/home/jiaxuzhu/developer/mxnet/python')
import os
import numpy as np
import scipy
import sys
sys.path.insert(0, '/home/jiaxuzhu/developer/mxnet/python')
import mxnet as mx
import cv2
import logging

def data_separation(data):
    n_data = len(data)
    train_data = data[:n_data/2, :]
    val_data = data[n_data/2:n_data/4*3, :]
    test_data = data[n_data/4*3:, :]
    return {'train': train_data, 'val': val_data, 'test': test_data}


class TextureClassifier:
    def __init__(self,  model_dir, prefix, n_iter, n_class):
        self.n_class = n_class
        self.iter = {}
        self.model_dir = model_dir
        self.init_model = mx.model.FeedForward.load(os.path.join(model_dir, prefix), n_iter, ctx=mx.gpu())
        internals = self.init_model.symbol.get_internals()
        symbol = internals['flatten_output']
        symbol = mx.symbol.FullyConnected(data=symbol, name='fullc', num_hidden=n_class)
        self.symbol = mx.symbol.SoftmaxOutput(data=symbol, name='softmax')
        self.net = None

    def mx_init(self, data_dir, b_size):
        sets = ['train', 'val', 'test']

        for dataset in sets:
            rec_file = os.path.join(data_dir, '%s.rec' % dataset)
            if not os.path.exists(rec_file):
                continue
            self.iter[dataset] = mx.io.ImageRecordIter(
                path_imgrec=rec_file,
                batch_size=b_size,
                data_shape=(3, 224, 224),
                mean_img=os.path.join(self.model_dir, 'mean_224.nd'),
            )

    def mx_training(self, n_epoch, l_rate, b_size, dst_prefix, ctx):
        opt = mx.optimizer.SGD(learning_rate=l_rate)
        self.net = mx.model.FeedForward(ctx=ctx, symbol=self.symbol, num_epoch=n_epoch, optimizer=opt,
                                        arg_params=self.init_model.arg_params, aux_params=self.init_model.aux_params,
                                        allow_extra_params=True)
        self.net.fit(self.iter['train'], eval_data=self.iter['val'],
                     batch_end_callback=mx.callback.Speedometer(b_size, 30),
                     epoch_end_callback=mx.callback.do_checkpoint(dst_prefix))

        self.net.save(dst_prefix)

    def mx_confusion(self, lst_file):
        prob = self.init_model.predict(self.iter['test'])
        logging.info('Finish predict...')
	
        count = 0
	idx = 0
        py = np.argmax(prob, axis=1)
        confusion = np.zeros((self.n_class, self.n_class)).astype('uint64')
	with open(lst_file, 'r') as lstFile:
	    for line in lstFile.readlines():
    		label = int(line.split('\t')[1])
		confusion[label, py[idx]] += 1
		if label == py[idx]:
		    count += 1
		idx += 1
	print float(count) / len(py)
        return confusion

    def mx_predict(self, data_path, b_size):
        test_iter = mx.io.ImageRecordIter(
            path_imgrec=data_path,
            batch_size=b_size,
            data_shape=(3, 224, 224),
            mean_img=os.path.join(self.model_dir, 'mean_224.nd'),
        )
        prob = self.init_model.predict(test_iter)
        return prob

    def mx_predict_np(self, data, b_size):
        mean_img = mx.nd.load(os.path.join(self.model_dir, 'mean_224.nd'))["mean_img"].asnumpy()
        mean_img = np.reshape(mean_img, (1, 3, 224, 224))
        mean_img = np.tile(mean_img, (len(data), 1, 1, 1))
        data = np.subtract(data, mean_img)
        test_iter = mx.io.NDArrayIter(data, batch_size=b_size)
        prob = self.init_model.predict(test_iter)
        return prob

