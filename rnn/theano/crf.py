import theano
import theano.tensor as T
import numpy as np

from rnn.initializers import orthogonal

def logsumexp(X, axis=None):
    x_max = T.max(X, axis=axis, keepdims=True)
    return T.log(T.sum(T.exp(X-x_max), axis=axis, keepdims=True)) + x_max

class CRF(object):
    def __init__(self, ins, labels, init=orthogonal, name=None, dtype=theano.config.floatX):
        w_trans = np.random.randn(labels, 1+ins, labels).astype(dtype)
        w_trans[:,0] = 0
        for i in xrange(labels):
            for j in xrange(labels):
                init(w_trans[i, 1:, j])
        w_init = np.random.randn(1+ins, labels).astype(dtype)
        w_init[0] = 0
        for i in xrange(labels):
            init(w_init[1:, i])
        self.w_trans = theano.shared(w_trans)
        self.w_init = theano.shared(w_init)

    @property
    def weights(self):
        return [self.w_trans, self.w_init]

    @weights.setter
    def weights(self, ws):
        self.w_trans.set_value(ws[0])
        self.w_init.set_value(ws[1])
        
    def logprob(self, Y, X, mask=None):
        inits = T.dot(X[0], self.w_init[1:]) + self.w_init[0]
        trans = T.dot(X[1:], self.w_trans[:,1:]) + self.w_trans[:,0]
        k,b = Y.shape
        mesh = T.mgrid[0:k-1,0:b]
        i,j = mesh[0], mesh[1]
        if mask is None:
            num = inits[j,Y[0]] + T.sum(trans[i,j,Y[:-1],Y[1:]])
        else:
            num = inits[j,Y[0]]*mask[0] + T.sum(trans[i,j,Y[:-1],Y[1:]]*mask[1:])
        if mask is None:
            part = logsumexp(inits, axis=-1) + T.sum(logsumexp(trans[i,j,Y[:-1]], axis=-1))
        else:
            part = logsumexp(inits, axis=-1)*mask[0] + T.sum(logsumexp(trans[i,j,Y[:-1]], axis=-1)*mask[1:])
        return num - part
            

