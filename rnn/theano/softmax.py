import theano
import theano.tensor as T

from rnn.initializers import orthogonal

def softmax(x, axis=-1):
    return T.exp(logsoftmax(x, axis=axis))

def logsoftmax(x, axis=-1):
    return x - logsumexp(x, axis=axis, keepdims=True)

def logsumexp(X, axis=None, keepdims=False):
    x_max = T.max(X, axis=axis, keepdims=keepdims)
    X = X - T.max(X, axis=axis, keepdims=True)
    return T.log(T.sum(T.exp(X), axis=axis, keepdims=keepdims)) + x_max


