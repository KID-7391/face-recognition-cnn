import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import conv


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):

        # init weight with 0
        self.W = theano.shared(
            np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        # init bias with 0
        self.b = theano.shared(
            np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # get the prediction probability for each label
        # use softmax as result
        self.p_pred = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # predict labels for each input
        self.pred = T.argmax(self.p_pred, axis=1)

        # get params
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_pred)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.pred.ndim:
            raise TypeError(
                'y should have the same shape as self.pred',
                ('y', y.type, 'pred', self.pred.type)
            )

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.pred, y))
        else:
            return NotImplementedError()

