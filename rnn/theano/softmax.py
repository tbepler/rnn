import theano
import theano.tensor as T

def softmax(x, axis=-1):
    x_shift = x - x.max(axis=axis, keepdims=True)
    e_x = T.exp(x_shift)
    return e_x / e_x.sum(axis=axis, keepdims=True)

def logsoftmax(x, axis=-1):
    x_shift = x - x.max(axis=axis, keepdims=True)
    scale = T.exp(x_shift).sum(axis=axis, keepdims=True)
    return x_shift - T.log(scale)

def logsumexp(X, axis=None, keepdims=False):
    x_max = T.max(X, axis=axis, keepdims=keepdims)
    X = X - T.max(X, axis=axis, keepdims=True)
    return T.log(T.sum(T.exp(X), axis=axis, keepdims=keepdims)) + x_max


