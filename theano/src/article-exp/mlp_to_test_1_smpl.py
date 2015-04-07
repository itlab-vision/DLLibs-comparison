import cPickle
import os
import sys
import time

import StringIO

import numpy

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression, load_data

class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input
        rng = numpy.random.RandomState(1234)

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
        if activation == theano.tensor.nnet.sigmoid:
            W_values *= 4

        W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [self.W, self.b]



class MLP(object):
    def __init__(self, input, n_in, n_hidden1, n_hidden2, n_out):
        self.hiddenLayer1 = HiddenLayer(
            input=input,
            n_in=n_in,
            n_out=n_hidden1,
            activation=T.tanh
        )

        self.hiddenLayer2 = HiddenLayer(
            input=self.hiddenLayer1.output,
            n_in=n_hidden1,
            n_out=n_hidden2,
            activation=T.tanh
        )

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer2.output,
            n_in=n_hidden2,
            n_out=n_out
        )

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        self.errors = self.logRegressionLayer.errors

        self.params = self.hiddenLayer1.params + self.hiddenLayer2.params + self.logRegressionLayer.params

def test_mlp(dataset = 'mnist.pkl.gz', batch_size = 128):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for testing
    n_test = test_set_x.get_value(borrow=True).shape[0]
    n_test_batches = n_test / batch_size

    ######################
    # LOAD ACTUAL MODEL #
    ######################
    print '... loading the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    # construct the MLP class
    classifier = MLP(
        input=x,
        n_in=28 * 28,
        n_hidden1=394,
        n_hidden2=196,
        n_out=10
    )

    f = open('/home/dmitry/Projects/DNN-develop/theano/results/mlp',"rb")
    classifier.hiddenLayer1.W.set_value(cPickle.load(f), borrow=True)
    classifier.hiddenLayer1.b.set_value(cPickle.load(f), borrow=True)

    classifier.hiddenLayer2.W.set_value(cPickle.load(f), borrow=True)
    classifier.hiddenLayer2.b.set_value(cPickle.load(f), borrow=True)

    classifier.logRegressionLayer.W.set_value(cPickle.load(f), borrow=True)
    classifier.logRegressionLayer.b.set_value(cPickle.load(f), borrow=True)

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    print('batch_size ' + str(batch_size))
    print('n_test ' + str(n_test))
    print('n_test_batches ' + str(n_test_batches))
    wtime = time.clock()
    test_losses = [test_model(i) for i in xrange(n_test_batches)]
    test_score = numpy.mean(test_losses)

    wtime = (time.clock() - wtime) / n_test * 1000.;
    print('for 1 sample needed ' + str(wtime) + ' msec')
    print('test score ' + str(test_score * 100.))

if __name__ == '__main__':
    test_mlp()
