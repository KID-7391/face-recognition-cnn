import mylib


class LogisticRegression:
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
