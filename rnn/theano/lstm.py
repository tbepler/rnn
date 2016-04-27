import theano
import theano.tensor as th
import numpy as np

from activation import fast_tanh, fast_sigmoid
from rnn.initializers import orthogonal

def step(ifog, y0, c0, wy, iact=fast_sigmoid, fact=fast_sigmoid, oact=fast_sigmoid, gact=fast_tanh
         , cact=fast_tanh, mask=None):
    m = y0.shape[1]
    ifog = ifog + th.dot(y0, wy)
    i = iact(ifog[:,:m])
    f = fact(ifog[:,m:2*m])
    o = oact(ifog[:,2*m:3*m])
    g = gact(ifog[:,3*m:])
    c = c0*f + i*g
    y = o*cact(c)
    if mask is not None:
        mask = mask.dimshuffle(0, 'x')
        y *= mask
        c *= mask
    return y, c

def split(w):
    m = w.shape[1]/4
    n = w.shape[0] - m - 1
    return w[0], w[1:n+1], w[n+1:]

def gates(bias, wx, x):
    if 'int' in x.dtype:
        k,b = x.shape
        x = x.flatten()
        ifog = wx[x] + bias
    else:
        k,b,n = x.shape
        x = x.reshape((k*b,n))
        ifog = th.dot(x, wx) + bias
    ifog = ifog.reshape((k,b,ifog.shape[1]))
    return ifog

def lstm(w, y0, c0, x, mask=None, op=theano.scan, **kwargs):
    b, wx, wy = split(w)
    ifog = gates(b, wx, x)
    if mask is not None:
        f = lambda g, m, yp, cp, wy: step(g, yp, cp, wy, mask=m, **kwargs)
        seqs = [ifog, mask]
    else:
        f = lambda g, yp, cp, wy: step(g, yp, cp, wy, **kwargs)
        seqs = ifog
    [y,c], updates = op(f, seqs, [y0, c0], non_sequences=wy)
    return y, c

def scanl(w, y0, c0, x, mask=None, **kwargs):
    return lstm(w, y0, c0, x, mask=mask, op=theano.scan, **kwargs)

def scanr(w, y0, c0, x, mask=None, **kwargs):
    op = lambda *args, **kwargs: theano.scan(*args, go_backwards=True, **kwargs)
    y, c = lstm(w, y0, c0, x, mask=mask, op=op, **kwargs)
    return y[::-1], c[::-1]

def foldl(w, y0, c0, x, **kwargs):
    b, wx, wy = split(w)
    ifog = gates(b, wx, x)
    f = lambda g, yp, cp, wy: step(g, yp, cp, wy, **kwargs)
    [y,c], updates = theano.foldl(f, ifog, [y0, c0], non_sequences=wy)
    return y, c

def foldr(w, y0, c0, x, **kwargs):
    b, wx, wy = split(w)
    ifog = gates(b, wx, x)
    f = lambda g, yp, cp, wy: step(g, yp, cp, wy, **kwargs)
    [y,c], updates = theano.foldr(f, ifog, [y0, c0], non_sequences=wy)
    return y, c

class LSTM(object):
    def __init__(self, ins, units, init=orthogonal, name=None, dtype=theano.config.floatX
                 , iact=fast_sigmoid, fact=fast_sigmoid, oact=fast_sigmoid, gact=fast_tanh
                 , cact=fast_tanh, forget_bias=3):
        w = np.random.randn(1+units+ins, 4*units).astype(dtype)
        w[0] = 0
        w[0,units:2*units] = forget_bias
        init(w[1:])
        self.weights = theano.shared(w, name=name)
        self.iact = iact
        self.fact = fact
        self.oact = oact
        self.gact = gact
        self.cact = cact

    def scanl(self, y0, c0, x, mask=None):
        return scanl(self.weights, y0, c0, x, mask=mask, iact=self.iact, fact=self.fact, oact=self.oact
                     , gact=self.gact, cact=self.cact)

    def scanr(self, y0, c0, x, mask=None):
        return scanr(self.weights, y0, c0, x, mask=mask, iact=self.iact, fact=self.fact, oact=self.oact
                     , gact=self.gact, cact=self.cact)

    def foldl(self, y0, c0, x, mask=None):
        return foldl(self.weights, y0, c0, x, mask=mask, iact=self.iact, fact=self.fact, oact=self.oact
                     , gact=self.gact, cact=self.cact)

    def foldr(self, y0, c0, x, mask=None):
        return foldr(self.weights, y0, c0, x, mask=mask, iact=self.iact, fact=self.fact, oact=self.oact
                     , gact=self.gact, cact=self.cact)

class BiLSTM(object):
    def __init__(self, ins, units, **kwargs):
        self.n = units
        self.lstml = LSTM(ins, units, **kwargs)
        self.lstmr = LSTM(ins, units, **kwargs)

    @property
    def weights(self): return self.lstml.weights + self.lsmtr.weights

    def __call__(self, x, mask=None):
        b = x.shape[1]
        y0 = th.zeros((b, self.n))
        c0 = th.zeros((b, self.n))
        yl, _ = self.lstml.scanl(y0, c0, x, mask=mask)
        yr, _ = self.lstmr.scanr(y0, c0, x, mask=mask)
        return th.concatenate([yl, yr], axis=yl.ndim-1)









