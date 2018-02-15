import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import conv


class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, activation=T.tanh, rng=None, params=None):

        self.input = input

        # sometimes we use trained parameters to init W and b
        # if use uninitialized parameters
        if params is None:
            W_bound = np.sqrt(6.0 / (n_in + n_out))
            if activation == T.nnet.sigmoid:
                W_bound *= 4
            W = theano.shared(
                np.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=(n_in, n_out)),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )

            b = theano.shared(
                value=np.zeros(
                    n_out, dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        else:
            W, b = params

        # init the parameters of this class
        self.W = W
        self.b = b

        # get output and params
        output = T.dot(input, self.W) + self.b
        if activation is None:
            self.output = output
        else:
            self.output = activation(output)

        self.params = [self.W, self.b]
