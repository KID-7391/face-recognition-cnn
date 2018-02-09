from ConvLayer import ConvLayer
from PoolLayer import PoolLayer
from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression
from PIL import Image
import pickle
import gzip
import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import conv


# save params
def save_params(param1, param2, param3, param4, param5, param6):
    write_file = open('params.pkl', 'wb')
    pickle.dump(param1, write_file, -1)
    pickle.dump(param2, write_file, -1)
    pickle.dump(param3, write_file, -1)
    pickle.dump(param4, write_file, -1)
    pickle.dump(param5, write_file, -1)
    pickle.dump(param6, write_file, -1)
    write_file.close()


###########################
# load data from dataset_path
# divide it into train_data,valid_data,test_data
###########################
def load_data(dataset_path):
    img = Image.open(dataset_path)
    img_ndarray = np.asarray(img, dtype='float64') / 256
    faces = np.empty((400, 2679))
    for i in range(20):
        for j in range(20):
            faces[i*20+j] = np.ndarray.flatten(img_ndarray[i*57:(i+1)*57, j*47:(j+1)*47])

    label = np.empty(400)
    for i in range(40):
        label[i*10:(i+1)*10] = i
    label = label.astype(np.int)

    # divide into 3 sets
    train_data = np.empty((320, 2679))
    valid_data = np.empty((40, 2679))
    test_data = np.empty((40, 2679))
    train_label = np.empty(320)
    valid_label = np.empty(40)
    test_label = np.empty(40)

    for i in range(40):
        train_data[i*8:i*8+8] = faces[i*10:i*10+8]
        valid_data[i] = faces[i*10+8]
        test_data[i] = faces[i*10+9]
        train_label[i * 8:i * 8 + 8] = label[i * 10:i * 10 + 8]
        valid_label[i] = label[i * 10 + 8]
        test_label[i] = label[i * 10 + 9]

    # share dataset
    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    train_x, train_y = shared_dataset(train_data, train_label)
    valid_x, valid_y = shared_dataset(valid_data, valid_label)
    test_x, test_y = shared_dataset(test_data, test_label)

    return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]


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


def cnn_train(learning_rate=0.05, n_epochs=200, dataset='olivettifaces.gif',
              nkerns=[5, 10], batch_size=40):
    ##############
    # part 1

    # random number generator
    rng = np.random.RandomState(58541)

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
        filter_shape=(nkerns[0], 1, 5, 5)
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
        filter_shape=(nkerns[1], nkerns[0], 5, 5)
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
        n_in=2000,
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
    params = [layer_C1.params, layer_S2.params] + [layer_C3.params, layer_S4.params] + layer_C5.params + layer_F6.params

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
        for minibatch_index in range(int(n_train_batches)):

            iter = (epoch - 1) * n_train_batches + minibatch_index
            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            #get current losses
            if (iter + 1) % valid_frequency == 0:
                valid_loss = [valid_model(i) for i in range(int(n_valid_batches))]
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
                        for i in range(int(n_test_batches))
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
