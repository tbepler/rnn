import theano.tensor as th

def softmax(x):
    ma = th.max(x, axis=x.ndim-1, keepdims=True)
    y = x - ma
    denom = ma + th.log(th.sum(th.exp(y), axis=x.ndim-1, keepdims=True))
    return th.exp(x - denom)
