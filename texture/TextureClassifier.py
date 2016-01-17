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
    def __init__(self, data_dir, p_labels, n_sample):
        self.patches = None
        self.labels = None
        self.n_class = len(p_labels)
        idx = 0

        for p_name in p_labels:
            patches = np.load(os.path.join(data_dir, '%s_patches.npy' % p_name))
            for patch in patches:
                print np.max(patch), np.min(patch)
                cv2.imshow('ImageWindow', patch)
                cv2.waitKey()

            patches = np.transpose(patches, (0, 3, 1, 2))
            shuffle = np.random.permutation(len(patches))
            patches = patches[shuffle[:n_sample], :]
            patches = data_separation(patches)

            labels = np.multiply(np.ones((n_sample, 1)), idx).astype(int)
            labels = data_separation(labels)

            if self.patches is None:
                self.patches = patches
                self.labels = labels
            else:
                for key in patches.keys():
                    self.patches[key] = np.vstack((self.patches[key], patches[key]))
                    self.labels[key] = np.vstack((self.labels[key], labels[key]))

            idx += 1
            print

    def mx_init(self, model_dir, prefix, n_iter):
        model = mx.model.FeedForward.load(os.path.join(model_dir, prefix), n_iter)
        self.init_model = model
        internals = model.symbol.get_internals()
        print self.n_class
        symbol = internals['flatten_output']
        symbol = mx.symbol.FullyConnected(data=symbol, name='fc1', num_hidden=self.n_class)
        self.symbol = mx.symbol.SoftmaxOutput(data=symbol, name='softmax')

        self.mxnet_mean = mx.nd.load(os.path.join(model_dir, 'mean_224.nd'))["mean_img"].\
            asnumpy().reshape((1, 3, 224, 224))
        for key in self.patches:
            self.patches[key] = self.patches[key] - np.tile(self.mxnet_mean, (len(self.patches[key]), 1, 1, 1))

    def mx_training(self, l_rate, b_size):
        opt = mx.optimizer.SGD(learning_rate=l_rate)
        self.net = mx.model.FeedForward(ctx=mx.gpu(), symbol=self.symbol, num_epoch=5, optimizer=opt,
                                        arg_params=self.init_model.arg_params, aux_params=self.init_model.aux_params,
                                        allow_extra_params=True)
        self.train_iter = mx.io.NDArrayIter(self.patches['train'], label=self.labels['train'].flatten(),
                                       shuffle=True, batch_size=b_size)

        self.val_iter = mx.io.NDArrayIter(self.patches['val'], shuffle=True,
                                     label=self.labels['val'].flatten(), batch_size=b_size)


        self.net.fit(self.train_iter, eval_data=self.val_iter, batch_end_callback=mx.callback.Speedometer(b_size, 10))

        self.train_iter.reset()
        labels = self.net.predict(self.train_iter)
        print labels[:10, :]
        labels = np.argmax(labels, axis=1)
        print labels
        print self.labels['train'].flatten()
        print float(len(np.where(labels == self.labels['train'].flatten())[0])) / len(labels)

        self.net.save('/home/jiaxuzhu/developer/CSD395/model/inception_texture')

    def mx_predict(self, model_dir, prefix, n_iter):
        model = mx.model.FeedForward.load(os.path.join(model_dir, prefix), n_iter, ctx=mx.gpu())
        test_iter = mx.io.NDArrayIter(self.patches['test'], label=self.labels['test'].flatten(),
                                      shuffle=True, batch_size=64)
        train_iter = mx.io.NDArrayIter(self.patches['train'], label=self.labels['train'].flatten(),
                                       shuffle=True, batch_size=64)
        labels = np.argmax(model.predict(train_iter), axis=1)
        print float(len(np.where(labels == self.labels['train'].flatten())[0])) / len(labels)