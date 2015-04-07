import os
import sys
import time
import cPickle

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])

        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))

        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]


def evaluate_lenet5(learning_rate=0.01, n_epochs=150,
                    dataset='mnist.pkl.gz',
                    nkerns=[28, 56], batch_size=128):
    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # LOAD ACTUAL MODEL #
    ######################
    print '... loading the model'

    layer0_input = x.reshape((batch_size, 1, 28, 28))

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(3, 3)
    )

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 8, 8),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        input=layer2_input,
        n_in=nkerns[1] * 2 * 2,
        n_out=200,
        activation=T.tanh
    )

    layer3 = LogisticRegression(input=layer2.output, n_in=200, n_out=10)

    f = open('/home/dmitry/Projects/DNN-develop/theano/results/LeNet5', 'rb')
    layer0.W.set_value(cPickle.load(f), borrow = True)
    layer0.b.set_value(cPickle.load(f), borrow = True)
    layer1.W.set_value(cPickle.load(f), borrow = True)
    layer1.b.set_value(cPickle.load(f), borrow = True)
    layer2.W.set_value(cPickle.load(f), borrow = True)
    layer2.b.set_value(cPickle.load(f), borrow = True)
    layer3.W.set_value(cPickle.load(f), borrow = True)
    layer3.b.set_value(cPickle.load(f), borrow = True)

    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print('batch_size ' + str(batch_size))
    print('n_test ' + str(n_test_batches * batch_size))
    print('n_test_batches ' + str(n_test_batches))
    wtime = time.clock()
    test_losses = [test_model(i) for i in xrange(n_test_batches)]
    test_score = numpy.mean(test_losses)

    wtime = (time.clock() - wtime) / (n_test_batches * batch_size) * 1000.;
    print('for 1 sample needed ' + str(wtime) + ' msec')
    print('test score ' + str(test_score * 100.))

if __name__ == '__main__':
    evaluate_lenet5()
