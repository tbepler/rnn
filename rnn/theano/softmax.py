import theano
import theano.tensor as T

from rnn.initializers import orthogonal

def softmax(x):
    ma = T.max(x, axis=x.ndim-1, keepdims=True)
    y = x - ma
    denom = ma + T.log(th.sum(th.exp(y), axis=x.ndim-1, keepdims=True))
    return th.exp(x - denom)

def logsoftmax(x, axis=-1):
    return x - logsumexp(x, axis=axis)

def logsumexp(X, axis=None):
    x_max = T.max(X, axis=axis, keepdims=True)
    return T.log(T.sum(T.exp(X-x_max), axis=axis, keepdims=True)) + x_max


