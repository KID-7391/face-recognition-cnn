import mylib
from mylib import T


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):

        self.input = input

        # sometimes we use trained parameters to init W and b
        # if use uninitialized parameters
        if W is None:
            W_bound = np.sqrt(6.0 / (n_in + n_out))
            if activation == T.nnet.sigmoid:
                W_bound *= 4
            W = theano.shared(
                np.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=n_in * n_out),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )

        if b is None:
            b = theano.shared(
                value=np.zeros(
                    n_out, dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

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
