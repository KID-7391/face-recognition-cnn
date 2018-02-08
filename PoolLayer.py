import mylib


class PoolLayer:
    def __init__(self, input, image_shape, poolsize=(2, 2)):

        # init bias
        self.b = theano.shared(
            value=np.zeros(
                filter_shape[0], dtype=theano.config.floatX
            ),
            borrow=True
        )

        # pool and save params
        pooled_out = pool_2d(input=input, ds=poolsize, ignore_border=True,
                             padding=same, mode='max')
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = self.b
