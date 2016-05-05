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

def flatten_left(shape):
    return [shape[0]*shape[1]] + list(shape[2:])

def lstm(w, y0, c0, x, mask=None, op=theano.scan, unroll=-1, **kwargs):
    b, wx, wy = split(w)
    n = x.shape[0]
    ifog = gates(b, wx, x)
    if mask is not None:
        f = lambda g, m, yp, cp, wy: step(g, yp, cp, wy, mask=m, **kwargs)
        seqs = [ifog, mask]
    else:
        f = lambda g, yp, cp, wy: step(g, yp, cp, wy, **kwargs)
        seqs = ifog
    if unroll > 1:
        def _unroll_step(*args):
            g = args[0]
            if mask is not None:
                m = args[1]
                args = args[1:]
            else:
                m = None
            yp, cp, wy = args[1], args[2], args[3]
            yp, cp = yp[-1], cp[-1]
            ys, cs = [], []
            for j in xrange(unroll):
                mj = m[j] if m is not None else None
                yp, cp = step(g[j], yp, cp, wy, mask=mj, **kwargs)
                ys.append(yp)
                cs.append(cp)
            return th.stack(ys), th.stack(cs)
        #align the sequences to mulitples of unroll
        pad = (unroll - n % unroll) % unroll
        ifog = th.concatenate([ifog, th.zeros((pad, ifog.shape[1], ifog.shape[2]), dtype=ifog.dtype)], axis=0)
        if mask is not None:
            mask = th.concatenate([mask, th.zeros((pad, mask.shape[1]), dtype=mask.dtype)], axis=0)
        #reshape the sequences into chunks of size unroll
        chunks = (n+pad) // unroll
        ifog = ifog.reshape((chunks, unroll, ifog.shape[1], ifog.shape[2]))
        if mask is not None:
            mask = mask.reshape((chunks, unroll, mask.shape[1]))
        #peel the list to align
        pad_y0, pad_c0 = th.tile(th.shape_padleft(y0), (unroll, 1, 1)), th.tile(th.shape_padleft(c0), (unroll, 1, 1))
        #have to unbroadcast the below or ifelse complains about type mismatch......
        #pad_y0, pad_c0 = th.unbroadcast(th.shape_padleft(y0), 0), th.unbroadcast(th.shape_padleft(c0), 0)
        if mask is not None:
            seqs = [ifog, mask]
        else:
            seqs = [ifog]
        [y, c], _ = op(_unroll_step, seqs, [pad_y0, pad_c0], non_sequences=wy)
        #flatten y and c chunks and truncate to n
        y = th.reshape(y, flatten_left(y.shape))[:n]
        c = th.reshape(c, flatten_left(c.shape))[:n]
        #compute and append peeled y and c
        #seqs_peel = [ifog[-peel:], mask[-peel:]] if mask is not None else [ifog[-peel:]]
        #[ypeel, cpeel], _ = op(f, seqs_peel, [y[-1], c[-1]], non_sequences=wy) 
        #y_with_peel, c_with_peel = th.concatenate([y, ypeel], axis=0), th.concatenate([c, cpeel], axis=0)
        #from theano.ifelse import ifelse
        #y, c = ifelse(peel > 0, (y_with_peel, c_with_peel), (y, c))
        #ypeel, cpeel = ifelse(peel > 0, (ypeel, cpeel), (pad_y0, pad_c0))
        #idxs = th.arange(peel, n, unroll)
        #y, c = ifelse(peel > 0, (th.concatenate([ypeel, y], axis=0), th.concatenate([cpeel, c], axis=0)), (y,c))
    else:
        [y, c], _ = op(f, seqs, [y0, c0], non_sequences=wy)
    return y, c

def scanl(w, y0, c0, x, mask=None, **kwargs):
    return lstm(w, y0, c0, x, mask=mask, op=theano.scan, **kwargs)

def scanr(w, y0, c0, x, mask=None, **kwargs):
    y, c = lstm(w, y0, c0, x[::-1], mask=mask, op=theano.scan, **kwargs)
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

    @property
    def units(self): return self.weights.shape[1]//4

    def scanl(self, y0, c0, x, mask=None, **kwargs):
        return scanl(self.weights, y0, c0, x, mask=mask, iact=self.iact, fact=self.fact, oact=self.oact
                     , gact=self.gact, cact=self.cact, **kwargs)

    def scanr(self, y0, c0, x, mask=None, **kwargs):
        return scanr(self.weights, y0, c0, x, mask=mask, iact=self.iact, fact=self.fact, oact=self.oact
                     , gact=self.gact, cact=self.cact, **kwargs)

    def foldl(self, y0, c0, x, mask=None, **kwargs):
        return foldl(self.weights, y0, c0, x, mask=mask, iact=self.iact, fact=self.fact, oact=self.oact
                     , gact=self.gact, cact=self.cact, **kwargs)

    def foldr(self, y0, c0, x, mask=None, **kwargs):
        return foldr(self.weights, y0, c0, x, mask=mask, iact=self.iact, fact=self.fact, oact=self.oact
                     , gact=self.gact, cact=self.cact, **kwargs)

class BiLSTM(object):
    def __init__(self, ins, units, **kwargs):
        self.n = units
        self.lstml = LSTM(ins, units, **kwargs)
        self.lstmr = LSTM(ins, units, **kwargs)

    @property
    def weights(self): return self.lstml.weights + self.lsmtr.weights

    def __call__(self, x, mask=None, **kwargs):
        b = x.shape[1]
        y0 = th.zeros((b, self.n))
        c0 = th.zeros((b, self.n))
        yl, _ = self.lstml.scanl(y0, c0, x, mask=mask, **kwargs)
        yr, _ = self.lstmr.scanr(y0, c0, x, mask=mask, **kwargs)
        return th.concatenate([yl, yr], axis=yl.ndim-1)









