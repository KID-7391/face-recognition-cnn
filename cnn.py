import mylib
import ConvLayer
import PoolLayer
import HiddenLayer
import LogisticRegression

###########################
# this function is used to train cnn
# made up of the following parts:
#   1.init random generator, load dataset and process it
#   2.build cnn:
#       input -> C1(conv) + S2(pool) + C3(conv) + S4(pool) +
#          C5(full_connected) + F6(logisticregression) -> ouput
#   3.define the rules of the algorithm, including loss function,
#       rules of training,validating,testing,and parameters updating rules
#   4.train cnn,find the most suitable parameters
###########################


def cnn_train(learning_rate=0.05,n_epochs=200,dataset='olivettifaces.gif',
              nkerns=[5, 10],batch_size=40):

    ##############
    # part 1

    # random number generator
    rng = np.random.RandomState(54354)

    # load data
    # x is input, y is label
    datasets = load_data(dataset)
    train_x, train_y = datasets[0]
    valid_x, valid_y = datasets[1]
    test_x, test_y = datasets[2]

    # calculate the num of batches
    n_train_batches = train_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_x.get_value(borrow=True).shape[0] / batch_size

    # x refers to face data, as input
    # y refer to label, as output
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    #
    ##############

    ##############
    # part 2

    # reshape x to a 4d tensor
    # input image is 57x47
    layer_C1_input = x.reshape((batch_size, 1, 57, 47))

    # C1:conv layer, kernel is 5x5
    # after convolution, feature maps are 53x43
    layer_C1 = ConvLayer(
        rng,
        input=layer_C1_input,
        image_shape=(batch_size, 1, 57, 47),
        filter_shape=(nkerns[0], 1, 5, 5)
    )

    # S2:pooling layer, kernel is 2x2, step is 2
    # after pooling, feature maps are 26x21
    layer_S2 = PoolLayer(
        input=layer_C1.output,
        image_shape=(batch_size, nkerns[0], 53, 43),
    )

    # C3:conv layer, kernel is 5x5
    # after convolution, feature maps are 22x17
    layer_C3 = ConvLayer(
        rng,
        input=layer_S2.output,
        image_shape=(batch_size, nkerns[0], 26, 21),
        filter_shape=(nkerns[1], nkerns[0], 5, 5)
    )

    # S4:pooling layer, kernel is 2x2, step is 2
    # after pooling, feature maps are 11x8
    layer_S4 = PoolLayer(
        input=layer_C3.output,
        image_shape=(batch_size, nkerns[1], 22, 17),
    )

    # C5:full_connected layer
    layer_C5_input = layer_S4.output.flatten(2)
    layer_C5 = HiddenLayer(
        rng,
        input=layer_C5_input,
        n_in=nkerns[1] * 11 * 8,
        n_out=2000,
    )

    # F6:logisticregression layer
    layer_F6 = LogisticRegression(
        input=layer_C5.output,
        n_in=layer_C5.n_out,
        n_out=40
    )

    #
    ##############

    ##############
    # part 3

    # loss function
    loss = layer_F6.negative_log_likelihood(y)

    # set model
    test_model = theano.function(
        [index],
        layer_F6.errors(y),
        givens={
            x: test_x[index * batch_size: (index + 1) * batch_size],
            y: test_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    valid_model = theano.function(
        [index],
        layer_F6.errors(y),
        givens={
            x: valid_x[index * batch_size: (index + 1) * batch_size],
            y: valid_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    #all params
    params = layer_C1.params + layer_S2.params + layer_C3.params + layer_S4.params + layer_C5.params + layer_F6.params

    #gradients of parameters
    grads = T.grad(loss, params)

    #update rule
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    #train model
    train_model = theano.function(
        [index],
        loss,
        updates=updates,
        givens={
            x: train_x[index * batch_size: (index + 1) * batch_size],
            y: train_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    #
    ##############

    ##############
    # part 4

    patience = 800
    patience_increase = 2
    improvement_threshold = 0.99
    valid_frequency = min(n_train_batches, patience / 2)

    best_valid_loss = np.inf
    best_iter = 0
    test_score = 0.0
    start_time = time.clock()

    epoch = 0
    done_looping = False

    #train cnn with minibatch SGD
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            cost_ij = train_model(minibatch_index)

            #get current losses
            if (iter + 1) % valid_frequency == 0:
                valid_loss = [valid_model(i) for i in xrange(n_valid_batches)]
                cur_valid_loss = np.mean(valid_loss)

                #if we got better params
                if cur_valid_loss < best_valid_loss:

                    #improve patience if improvement is good enought
                    if cur_valid_loss < best_valid_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    #save best iteration number
                    best_valid_loss = cur_valid_loss
                    best_iter = iter
                    save_params(
                        layer_C1.params, layer_S2.params,
                        layer_C3.params, layer_S4.params,
                        layer_C5.params, layer_F6.params
                    )

                    #test it on the test set
                    test_loss = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = np.mean(test_loss)

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()

    #
    ##############


if __name__ == '__main__':
    cnn_train()
