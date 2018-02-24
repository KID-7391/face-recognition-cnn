from ConvLayer import ConvLayer
from PoolLayer import PoolLayer
from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression
from PIL import Image
import pickle
import gzip
import os

import numpy as np

import theano
import theano.tensor as T


n_subject = 9
n_test_data = 10
hight_filter = 5
width_filter = 5
hight_image = 64
width_image = 64
size_image = hight_image * width_image


# load params
def load_params(params_file):
    read_file = open(params_file, 'rb')
    params1 = pickle.load(read_file)
    params2 = pickle.load(read_file)
    params3 = pickle.load(read_file)
    params4 = pickle.load(read_file)
    params5 = pickle.load(read_file)
    params6 = pickle.load(read_file)
    read_file.close()
    return params1, params2, params3, params4, params5, params6

# load testdata
def load_data(dataset_path):

    # load mydata
    if dataset_path == 'my_testdata':
        faces = np.empty((n_subject * n_test_data, 64*64))
        label = np.empty(n_subject * n_test_data)
        name = []
        cnt = 0
        for subject in os.listdir('my_testdata'):
            name.append(subject)
            ls = os.listdir('my_testdata' + '/' + subject)
            for i in ls:
                img = Image.open('my_testdata' + '/' + subject + '/' +i)
                img_ndarray = np.asarray(img, dtype='float64') / 256
                faces[cnt] = np.ndarray.flatten(img_ndarray)
                label[cnt] = cnt / n_test_data
                cnt += 1

    label = label.astype(np.int)

    return faces, label, name



###########################
# this function use trained cnn to recognize faces
# made up of the following parts:
#   1.load test data and trained params
#   2.build cnn like we did in cnn_train function
#   3.input faces into cnn and output results
###########################

def cnn_use(dataset, nkerns=[10, 20]):
    ##############
    # part 1
    # load testdata
    faces, label, name = load_data(dataset)

    # load params
    params = load_params('params.pkl')

    #
    ##############

    ##############
    # part 2
    x = T.matrix('x')
    total_data = n_subject * n_test_data


    # reshape x to a 4d tensor
    # input image is hight_image x width_image
    hight_input = hight_image
    width_input = width_image
    layer_C1_input = x.reshape((total_data, 1, hight_image, width_image))

    # C1:conv layer, kernel is hight_filter x weight_filter
    # after convolution, feature maps are
    # (hight_input-hight_filter+1) x (width_input-width_filter+1)
    layer_C1 = ConvLayer(
        input=layer_C1_input,
        image_shape=(total_data, 1, hight_input, width_input),
        filter_shape=(nkerns[0], 1, hight_filter, width_filter),
        W=params[0]
    )
    hight_input = (hight_input - hight_filter + 1)
    width_input = (width_input - width_filter + 1)

    # S2:pooling layer, kernel is 2x2, step is 2
    # after pooling, feature maps are input_shape / 2
    layer_S2 = PoolLayer(
        input=layer_C1.output,
        filter_shape=(nkerns[0], 1, hight_filter, width_filter),
        b=params[1]
    )
    hight_input = int(hight_input / 2)
    width_input = int(width_input / 2)

    # C3:conv layer
    layer_C3 = ConvLayer(
        input=layer_S2.output,
        image_shape=(total_data, nkerns[0], hight_input, width_input),
        filter_shape=(nkerns[1], nkerns[0], hight_filter, width_filter),
        W=params[2]
    )
    hight_input = (hight_input - hight_filter + 1)
    width_input = (width_input - width_filter + 1)

    # S4:pooling layer, kernel is 2x2, step is 2
    # after pooling, feature maps are 11x8
    layer_S4 = PoolLayer(
        input=layer_C3.output,
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        b=params[3]
    )
    hight_input = int(hight_input / 2)
    width_input = int(width_input / 2)

    # C5:full_connected layer
    layer_C5_input = layer_S4.output.flatten(2)
    layer_C5 = HiddenLayer(
        input=layer_C5_input,
        n_in=nkerns[1] * hight_input * width_input,
        n_out=2000,
        params=params[4]
    )

    # F6:logisticregression layer
    layer_F6 = LogisticRegression(
        input=layer_C5.output,
        n_in=2000,
        n_out=n_subject,
        params=params[5]
    )

    # define a function f to predict
    f = theano.function([x], layer_F6.pred)

    #
    ##############

    ##############
    # part 3
    # get result
    # print error images and accracy
    result = f(faces)
    errors = 0
    for i in range(total_data):
        if result[i] != label[i]:
            img = Image.fromarray(256*faces[i].reshape([64, 64]))
            img.show()
            print('Recognize', name[label[i]], 'as', name[result[i]])
            errors += 1

    print('Accracy :', str(int(1000 * (1 - errors / total_data))/10)+'%')


    #
    ##############

if __name__ == '__main__':
    cnn_use('my_testdata')


