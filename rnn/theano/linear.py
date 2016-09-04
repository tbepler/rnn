import theano
import theano.tensor as T
import numpy as np

from rnn.theano.softmax import logsoftmax
from rnn.initializers import orthogonal

class Linear(object):
    def __init__(self, n_in, n_out, init=orthogonal, init_bias=0, dtype=theano.config.floatX, random=np.random, name=None):
        w = random.randn(n_in+1, n_out).astype(dtype)
        w[0] = init_bias
        init(w[1:])
        self.ws = theano.shared(w, name=name)
        self.name = name

    @property
    def weights(self):
        return [self.ws]
    
    def __getstate__(self):
        state = {}
        state['weights'] = self.ws.get_value(borrow=True)
        state['name'] = self.name
        return state

    def __setstate__(self, state):
        self.name = state['name']
        self.ws = theano.shared(state['weights'], borrow=True)

    def __call__(self, x):
        return T.dot(x, self.ws[1:]) + self.ws[0]

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
    
