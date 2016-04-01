import theano
import theano.tensor as T
import numpy as np

from rnn.initializers import orthogonal

class Linear(object):
    def __init__(self, n_in, n_out, init=orthogonal, dtype=theano.config.floatX, name=None):
        w = np.random.randn(n_in+1, n_out).astype(dtype)
        w[0] = 0
        init(w[1:])
        self.weights = theano.shared(w, name=name)

    def __call__(self, x):
        return T.dot(x, self.weights[1:]) + self.weights[0]
        #dims = x.shape
        #x = x.T.flatten(2)
        #m = T.prod(dims[:-1])
        #n = dims[-1]
        #x = x.reshape((m, n))
        #y = T.dot(x, self.weights[1:]) + self.weights[0]
        #ydims = dims[:-1] + (self.weights.shape[1],)
        #return y.reshape(ydims)

if __name__=='__main__':
    lin = Linear(5, 10)
    x = T.tensor3()
    y = lin(x)
    f = theano.function([x], y)
    k,b = 100, 10
    x = np.random.randn(k, b, 5)
    f(x)
    
