import theano
import theano.tensor as T
import numpy as np

from softmax import logsoftmax
from rnn.initializers import orthogonal

def logsumexp(X, axis=None):
    x_max = T.max(X, axis=axis, keepdims=True)
    return T.log(T.sum(T.exp(X-x_max), axis=axis, keepdims=True)) + x_max

class CRF(object):
    def __init__(self, ins, labels, init=orthogonal, name=None, dtype=theano.config.floatX):
        w_trans = np.random.randn(labels, 1+ins, labels).astype(dtype)
        w_trans[:,0] = 0
        for i in xrange(labels):
            init(w_trans[i, 1:])
        w_init = np.random.randn(1+ins, labels).astype(dtype)
        w_init[0] = 0
        init(w_init[1:])
        self.w_trans = theano.shared(w_trans)
        self.w_init = theano.shared(w_init)

    @property
    def weights(self):
        return [self.w_trans, self.w_init]

    @weights.setter
    def weights(self, ws):
        self.w_trans.set_value(ws[0])
        self.w_init.set_value(ws[1])
        
    def logprob(self, Y, X):
        inits = T.dot(X[0], self.w_init[1:]) + self.w_init[0]
        trans = T.dot(X[1:], self.w_trans[:,1:]) + self.w_trans[:,0]
        k,b = Y.shape
        mesh = T.mgrid[0:k-1,0:b]
        i,j = mesh[0], mesh[1]
        num = T.concatenate([T.shape_padleft(inits), trans[i,j,Y[:-1]]], axis=0)
        return logsoftmax(num, axis=-1)
            

