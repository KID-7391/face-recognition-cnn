import mylib

#filter_shape:(number of filters, num input feature maps,filter height, filter width)
#image_shape:(batch size, num input feature maps,image height, image width)

class ConvLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):

        #check parameters
        assert  image_shape[1] == filter_shape[1]
        self.input = input

        #calculate fan_in and fan_out to init weight

        #fan_in = num input feature maps * filter_height * filter_width
        fan_in = np.prod(filter_shape[1:])

        #fan_out = num output feature maps * filter_height * filter_width / pool_size
        fan_out = image_shape[1] * np.prod(filter_shape[2:]) / np.prod(image_shape)

        #init weigth with uniformly within the interval [−b,b]
        #b = sqrt(6 / (fan_in + fan_out))
        W_bound = sqrt(6.0 / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low = -W_bound, high = W_bound, size=filter_shape),
                dtype=theano.config.floatX
                ),
                borrow=True
        )

        #init bias
        self.b = theano.shared(
            value=np.zeros(
                filter_shape[0], dtype=theano.config.floatX
            ),
            borrow=True
        )

        #do convolution
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        #get output and params
        self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.params = [self.W, self.b]
