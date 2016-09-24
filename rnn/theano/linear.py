import theano
import theano.tensor as T
import numpy as np

from rnn.theano.softmax import logsoftmax
from rnn.initializers import orthogonal

class Linear(object):
    def __init__(self, n_in, n_out, init=orthogonal, init_bias=0, dtype=theano.config.floatX, random=np.random, name=None, use_bias=True):
        if use_bias:
            w = random.randn(n_in+1, n_out).astype(dtype)
            w[0] = init_bias
            init(w[1:])
        else:
            w = random.randn(n_in, n_out).astype(dtype)
            init(w)
        self.use_bias = use_bias
        self.ws = theano.shared(w, name=name)
        self.name = name

    @property
    def shared(self):
        return [self.ws]

    @property
    def weights(self):
        if self.use_bias:
            return [self.ws[1:]]
        return [self.ws]

    @property
    def bias(self):
        if self.use_bias:
            return [self.ws[0]]
        return []
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['ws'] = self.ws.get_value(borrow=True)
        return state

    def __setstate__(self, state):
        if 'ws' in state:
            self.__dict__.update(state)
            self.ws = theano.shared(self.ws, borrow=True) 
        else:
            self.name = state['name']
            self.ws = theano.shared(state['weights'], borrow=True)
            self.use_bias = True #just take a guess! ...

    def __call__(self, x):
        if self.use_bias:
            return T.dot(x, self.ws[1:]) + self.ws[0]
        else:
            return T.dot(x, self.ws)

class LinearDecoder(object):
    def __init__(self, n_in, n_out, init=orthogonal, dtype=theano.config.floatX, name=None):
        w = np.random.randn(n_in+1, n_out).astype(dtype)
        w[0] = 0
        init(w[1:])
        self.ws = theano.shared(w, name=name)

    def __call__(self, x):
        return T.dot(x, self.ws[1:]) + self.ws[0]

    @property
    def weights(self):
        return [self.ws]

    @weights.setter
    def weights(self, ws):
        self.ws.set_value(ws[0])
        
    def logprob(self, Y, X):
        Yh = logsoftmax(T.dot(X, self.ws[1:]) + self.ws[0], axis=-1)
        return Yh 

if __name__=='__main__':
    lin = Linear(5, 10)
    x = T.tensor3()
    y = lin(x)
    f = theano.function([x], y)
    k,b = 100, 10
    x = np.random.randn(k, b, 5)
    f(x)
    
