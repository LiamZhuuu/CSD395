import os
import numpy as np
import mxnet as mx

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
            patches = np.transpose(patches, (0, 3, 1, 2))
            shuffle = np.random.permutation(len(patches))
            patches = patches[shuffle[:n_sample], :]
            patches = data_separation(patches)

            labels = np.multiply(np.ones((n_sample, 1)), idx).astype(int)
            print np.max(labels)
            labels = data_separation(labels)

            if self.patches is None:
                self.patches = patches
                self.labels = labels
            else:
                for key in patches.keys():
                    self.patches[key] = np.vstack((self.patches[key], patches[key]))
                    self.labels[key] = np.vstack((self.labels[key], labels[key]))

            idx += 1

    def mx_init(self, model_dir, prefix, n_iter):
        model = mx.model.FeedForward.load(os.path.join(model_dir, prefix), n_iter)
        self.init_model = model
        internals = model.symbol.get_internals()
        symbol = internals['fc7_output']
        symbol = mx.symbol.FullyConnected(data=symbol, name='fc8', num_hidden=self.n_class)
        self.symbol = mx.symbol.SoftmaxOutput(data=symbol, name='softmax')

    def mx_training(self, l_rate, b_size):
        opt = mx.optimizer.sgd()
        self.net = mx.model.FeedForward(ctx=mx.gpu(), symbol=self.symbol,
                                        arg_params=self.init_model.arg_params, aux_params=self.init_model.aux_params,
                                        allow_extra_params=True)
        train_iter = mx.io.NDArrayIter(self.patches['train'], label=self.labels['train'], batch_size=b_size)
        val_iter = mx.io.NDArrayIter(self.patches['val'], label=self.labels['val'], batch_size=b_size)
        test_iter = mx.io.NDArrayIter(self.patches['test'], label=self.labels['test'], batch_size=b_size)
        self.net.fit(train_iter, eval_data=val_iter)

