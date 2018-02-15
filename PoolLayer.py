import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import conv


class PoolLayer(object):
    def __init__(self, input, filter_shape, poolsize=(2, 2), b=None):

        # if no params avilable
        if b is None:
            # init bias
            self.b = theano.shared(
                value=np.zeros(
                    filter_shape[0], dtype=theano.config.floatX
                ),
                borrow=True
            )
        # else init b with params
        else:
            self.b = b


        # pool and save params
        pooled_out = pool_2d(input=input, ds=poolsize, ignore_border=True,
                            mode='max')

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = self.b
