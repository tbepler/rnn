import sys
import theano
import theano.tensor as th
import numpy as np
from theano.tensor.nnet.bn import batch_normalization

sys.path.append("../../")

class BatchNormalizer(object):
    def __init__(self, n_in):
        self.gamma = theano.shared(value = np.ones((n_in), dtype=theano.config.floatX), name='gamma')
        self.beta = theano.shared(value = np.zeros((n_in), dtype=theano.config.floatX), name='beta')

    def weights(self): return [self.gamma, self.beta]

    def scan(self, inputs):
        gamma = self.gamma.get_value()
        beta = self.beta.get_value()
        bn_output = batch_normalization(inputs = inputs, gamma = self.gamma, beta = self.beta, mean = inputs.mean((1,2), keepdims=True), std = th.maximum(inputs.std((1,2), keepdims=True), 0.1**4), mode='high_mem')
        return bn_output

    def delete(self, del_units, history):
        gamma = self.gamma.get_value()
        beta = self.beta.get_value()
        print "Gamma: %s Beta: %s" % (gamma, beta)
        for i in del_units:
            gamma = np.delete(gamma, i, 0)
            beta = np.delete(beta, i, 0)
        for hist in history:
            hist_gamma = hist[0].get_value()
            hist_beta = hist[1].get_value()
            for i in del_units:
                hist_gamma = np.delete(hist_gamma, i, 0)
                hist_beta = np.delete(hist_beta, i, 0)
            hist[0].set_value(hist_gamma)
            hist[1].set_value(hist_beta)
        self.gamma.set_value(gamma)
        self.beta.set_value(beta)

    def add(self, add_units, history):
        gamma = self.gamma.get_value()
        beta = self.beta.get_value()
        print "Gamma: %s Beta: %s" % (gamma, beta)
        gamma = np.insert(gamma, -1, np.ones((add_units,), dtype=theano.config.floatX), 0)
        beta = np.insert(beta, -1, np.zeros((add_units,), dtype=theano.config.floatX), 0)
        for hist in history:
            hist_gamma = hist[0].get_value()
            hist_beta = hist[1].get_value()
            hist_gamma = np.insert(hist_gamma, -1, np.ones((add_units,), dtype=theano.config.floatX), 0)
            hist_beta = np.insert(hist_beta, -1, np.zeros((add_units,), dtype=theano.config.floatX), 0)
            hist[0].set_value(hist_gamma)
            hist[1].set_value(hist_beta)
        self.gamma.set_value(gamma)
        self.beta.set_value(beta)
