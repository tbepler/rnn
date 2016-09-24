from __future__ import division

import theano
import theano.tensor as th
import numpy as np

from rnn.theano.activation import fast_tanh, fast_sigmoid
from rnn.initializers import orthogonal

def step(ifog, y0, c0, wy, iact=fast_sigmoid, fact=fast_sigmoid, oact=fast_sigmoid, gact=fast_tanh
        , cact=fast_tanh, mask=None, activation=lambda x: x, clip=None):
    m = y0.shape[-1]
    ifog = ifog + th.dot(y0, wy)
    i = iact(ifog.T[:m].T)
    f = fact(ifog.T[m:2*m].T)
    o = oact(ifog.T[2*m:3*m].T)
    g = gact(ifog.T[3*m:].T)
    c = c0*f + i*g
    if clip is not None:
        c = theano.gradient.grad_clip(c, -clip, clip)
    y = activation(o*cact(c))
    if mask is not None:
        #I = (1-mask).nonzero()
        #y = th.set_subtensor(y[I], y0[I])
        #c = th.set_subtensor(c[I], c0[I])
        if mask.ndim < y.ndim:
            mask = th.shape_padright(mask)
        y = y*mask + y0*(1-mask)
        c = c*mask + c0*(1-mask)
    return y, c

def split(w):
    m = w.shape[1]//4
    n = w.shape[0] - m - 1
    return w[0], w[1:n+1], w[n+1:]

def gates(bias, wx, x):
    if 'int' in x.dtype:
        ifog = wx[x] + bias
    else:
        ifog = th.dot(x, wx) + bias
    return ifog

def flatten_left(shape):
    return [shape[0]*shape[1]] + list(shape[2:])

def unfoldl(w, y0, c0, x, steps, mask=None, **kwargs):
    b, wx, wy = split(w)
    ifog = gates(b, wx, x)
    if mask is None:
        def _step(i, yp, cp, g, wy):
            return step(g, yp, cp, wy, **kwargs)
        s = T.arange(steps)
    else:
        def _step(m, yp, cp, g, wy):
            return step(g, yp, cp, wy, mask=m, **kwargs)
        s = mask[:steps]
    res, _ = theano.scan(_step, [s], outputs_info=[y0, c0], non_sequences=[g,wy])
    return res

def unfoldr(w, y0, c0, x, steps, mask=None, **kwargs):
    if mask is not None:
        mask = mask[::-1]
    y,c = unfoldl(w, y0, c0, x, steps, mask=mask, **kwargs)
    return y[::-1], c[::-1]

def lstm(w, y0, c0, x, mask=None, op=theano.scan, unroll=-1, scan_kwargs={}, **kwargs):
    b, wx, wy = split(w)
    n = x.shape[0]
    ifog = gates(b, wx, x)
    if mask is not None:
        f = lambda g, m, yp, cp, wy: step(g, yp, cp, wy, mask=m, **kwargs)
        seqs = [ifog, mask]
    else:
        f = lambda g, yp, cp, wy: step(g, yp, cp, wy, **kwargs)
        seqs = ifog
    if unroll > 0:
        #perform loop in the fixed length unroll size - in other words, inputs MUST be of this length
        ys = []
        cs = []
        #Y = th.zeros((unroll,)+y0.shape)
        #C = th.zeros((unroll,)+c0.shape)
        for i in range(unroll):
            if mask is not None:
                m = mask[i]
            else:
                m = None
            yi, ci = step(ifog[i], y0, c0, wy, mask=m, **kwargs)
            ys.append(yi)
            cs.append(ci)
            #Y = th.set_subtensor(Y[i], yi)
            #C = th.set_subtensor(C[i], ci)
            y0, c0 = yi, ci
        y, c = th.stack(ys, axis=0), th.stack(cs, axis=0) 
    else:
        #from theano.compile.nanguardmode import NanGuardMode
        #mode = NanGuardMode(nan_is_error=True, inf_is_error=False, big_is_error=False)
        #[y, c], _ = op(f, seqs, [y0, c0], non_sequences=wy, mode=mode)
        [y, c], _ = op(f, seqs, [y0, c0], non_sequences=wy, **scan_kwargs)
    return y, c

def scanl(w, y0, c0, x, mask=None, **kwargs):
    return lstm(w, y0, c0, x, mask=mask, op=theano.scan, **kwargs)

def scanr(w, y0, c0, x, mask=None, **kwargs):
    if mask is not None:
        mask = mask[::-1]
    y, c = lstm(w, y0, c0, x[::-1], mask=mask, op=theano.scan, **kwargs)
    return y[::-1], c[::-1]

def foldl(w, y0, c0, x, mask=None, **kwargs):
    y, c = scanl(w, y0, c0, x, mask=mask, **kwargs)
    return y[-1], c[-1]

def foldr(w, y0, c0, x, mask=None, **kwargs):
    y, c = scanr(w, y0, c0, x, mask=mask, **kwargs)
    return y[0], c[0]

class LSTM(object):
    def __init__(self, ins, units, init=orthogonal, name=None, dtype=theano.config.floatX
                 , iact=fast_sigmoid, fact=fast_sigmoid, oact=fast_sigmoid, gact=fast_tanh
                 , cact=fast_tanh, forget_bias=3, random=np.random):
        w = random.randn(1+units+ins, 4*units).astype(dtype)
        w[0] = 0
        w[0,units:2*units] = forget_bias
        init(w[1:])
        self.ws = theano.shared(w, name=name, borrow=True)
        #self.c0 = theano.shared(np.zeros(units,dtype=dtype),borrow=True)
        self.c0 = th.zeros(units, dtype=dtype)
        #self.y0 = theano.shared(np.zeros(units,dtype=dtype),borrow=True)
        self.y0 = th.zeros(units, dtype=dtype)
        self.name = name
        self.iact = iact
        self.fact = fact
        self.oact = oact
        self.gact = gact
        self.cact = cact

    @property
    def shared(self): return [self.ws]

    @property
    def weights(self): return [self.ws[1:]]

    @property
    def bias(self): return [self.ws[0]]

    @property
    def units(self): return self.ws.get_value(borrow=True).shape[1]//4

    def __getstate__(self):
        state = {}
        state['name'] = self.name
        state['weights'] = self.ws.get_value(borrow=True)
        #state['c0'] = self.c0.get_value(borrow=True)
        #state['y0'] = self.y0.get_value(borrow=True)
        state['activations'] = [self.iact, self.fact, self.oact, self.gact, self.cact]
        return state

    def __setstate__(self, state):
        self.name = state['name']
        self.ws = theano.shared(state['weights'], borrow=True)
        #self.c0 = theano.shared(state['c0'], borrow=True)
        self.c0 = th.zeros(state['weights'].shape[1]//4, dtype=self.ws.dtype)
        #self.y0 = theano.shared(state['y0'], borrow=True)
        self.y0 = th.zeros(state['weights'].shape[1]//4, dtype=self.ws.dtype)
        acts = state['activations']
        self.iact = acts[0]
        self.fact = acts[1]
        self.oact = acts[2]
        self.gact = acts[3]
        self.cact = acts[4]

    def build_init_states(self, x, y0, c0):
        if 'int' in x.dtype:
            b = x.shape[1]
            #shape = x.shape[1:]
        else:
            #shape = x.shape[1:-1]
            b = x.shape[1]
        #z = th.shape_padright(th.zeros(shape))
        if y0 is None:
            y0 = th.tile(th.shape_padleft(self.y0), (b,1))
            #y0 = self.y0
        if c0 is None:
            c0 = th.tile(th.shape_padleft(self.c0), (b,1))
            #c0 = self.c0
        return y0, c0

    def scanl(self, x, y0=None, c0=None, mask=None, scan_kwargs={}, **kwargs):
        y0, c0 = self.build_init_states(x, y0, c0)
        return scanl(self.ws, y0, c0, x, mask=mask, iact=self.iact, fact=self.fact, oact=self.oact
                     , gact=self.gact, cact=self.cact, scan_kwargs=scan_kwargs, **kwargs)

    def scanr(self, x, y0=None, c0=None, mask=None, scan_kwargs={}, **kwargs):
        y0, c0 = self.build_init_states(x, y0, c0)
        return scanr(self.ws, y0, c0, x, mask=mask, iact=self.iact, fact=self.fact, oact=self.oact
                     , gact=self.gact, cact=self.cact, scan_kwargs={}, **kwargs)

    def foldl(self, x, y0=None, c0=None, mask=None, **kwargs):
        if y0 is None:
            #y0 = self.cact(self.y0)
            y0 = th.zeros(x.shape[1:]) + self.y0
        if c0 is None:
            c0 = th.zeros(x.shape[1:]) + self.c0
        return foldl(self.ws, y0, c0, x, mask=mask, iact=self.iact, fact=self.fact, oact=self.oact
                     , gact=self.gact, cact=self.cact, **kwargs)

    def foldr(self, x, y0=None, c0=None, mask=None, **kwargs):
        if y0 is None:
            #y0 = self.cact(self.y0)
            y0 = th.zeros(x.shape[1:]) + self.y0
        if c0 is None:
            c0 = th.zeros(x.shape[1:]) + self.c0
        return foldr(self.ws, y0, c0, x, mask=mask, iact=self.iact, fact=self.fact, oact=self.oact
                     , gact=self.gact, cact=self.cact, **kwargs)

    def unfoldl(self, x, steps, y0=None, c0=None, **kwargs):
        if y0 is None:
            #y0 = self.cact(self.y0)
            y0 = th.zeros(x.shape) + self.y0
        if c0 is None:
            c0 = th.zeros(x.shape) + self.c0
        return unfoldl(self.ws, y0, c0, x, steps, iact=self.iact, fact=self.fact, oact=self.oact
                , gact=self.gact, cact=self.cact, **kwargs)

    def unfoldr(self, x, steps, y0=None, c0=None, **kwargs):
        if y0 is None:
            #y0 = self.cact(self.y0)
            y0 = th.zeros(x.shape) + self.y0
        if c0 is None:
            c0 = th.zeros(x.shape) + self.c0
        return unfoldr(self.ws, y0, c0, x, steps, iact=self.iact, fact=self.fact, oact=self.oact
                , gact=self.gact, cact=self.cact, **kwargs)

class LayeredLSTM(object):
    def __init__(self, ins, layers, **kwargs):
        self.lstms = []
        for n in layers:
            self.lstms.append(LSTM(ins, n, **kwargs))
            ins = n

    @property
    def units(self):
        return sum(lstm.units for lstm in self.lstms)

    @property
    def weights(self):
        ws = []
        for lstm in self.lstms:
            ws.extend(lstm.weights)
        return ws

    def split(self, vector):
        if vector is not None:
            splits = []
            i = 0
            for lstm in lstms:
                n = lstm.units
                splits.append(vector.T[i:i+n].T)
                i += n
            return splits
        else:
            return [None for _ in self.lstms]

    def scanl(self, x, y0=None, c0=None, mask=None, **kwargs):
        y0, c0 = self.split(y0), self.split(c0)
        ys, cs = [], []
        for i in xrange(len(self.lstms)):
            lstm = self.lstms[i]
            y, c = lstm.scanl(x, y0=y0[i], c0=c0[i], mask=mask, **kwargs)
            ys.append(y[-1])
            cs.append(c[-1])
            x = y
        ys = th.concatenate(ys, axis=-1)
        cs = th.concatenate(cs, axis=-1)
        return y, c, ys, cs

    def scanr(self, x, y0=None, c0=None, mask=None, **kwargs):
        y0, c0 = self.split(y0), self.split(c0)
        ys, cs = [], []
        for i in xrange(len(self.lstms)):
            lstm = self.lstms[i]
            y, c = lstm.scanr(x, y0=y0[i], c0=c0[i], mask=mask, **kwargs)
            ys.append(y[-1])
            cs.append(c[-1])
            x = y
        ys = th.concatenate(ys, axis=-1)
        cs = th.concatenate(cs, axis=-1)
        return y, c, ys, cs

class BLSTM(object):
    def __init__(self, ins, units, **kwargs):
        self.nl = units // 2 + units % 2
        self.nr = units // 2
        self.lstml = LSTM(ins, self.nl, **kwargs)
        self.lstmr = LSTM(ins, self.nr, **kwargs)

    @property
    def weights(self): return self.lstml.weights + self.lstmr.weights

    @property
    def units(self): return self.nl + self.nr

    def __call__(self, x, mask=None, **kwargs):
        return self.scan(x, mask=mask, **kwargs)

    def scan(self, x, mask=None, **kwargs):
        yl, _ = self.lstml.scanl(x, mask=mask, **kwargs)
        yr, _ = self.lstmr.scanr(x, mask=mask, **kwargs)
        return th.concatenate([yl, yr], axis=yl.ndim-1)
    
    def fold(self, x, mask=None, **kwargs):
        yl, _ = self.lstml.foldl(x, mask=mask, **kwargs)
        yr, _ = self.lstmr.foldr(x, mask=mask, **kwargs)
        return th.concatenate([yl, yr], axis=yl.ndim-1)

    def unfold(self, x, steps, **kwargs):
        yl, _ = self.lstml.unfoldl(x, steps, **kwargs)
        yr, _ = self.lstmr.unfoldr(x, steps, **kwargs)
        return th.concatenate([yl,yr], axis=yl.ndim-1)

class LayeredBLSTM(object):
    def __init__(self, ins, layers, **kwargs):
        self.blstms = []
        for n in layers:
            self.blstms.append(BLSTM(ins, n, **kwargs))
            ins = n

    @property
    def weights(self): 
        return [w for blstm in self.blstms for w in blstm.weights]

    @property
    def units(self): return sum(blstm.units for blstm in self.blstms)

    def scan(self, x, mask=None, **kwargs):
        for blstm in self.blstms:
            x = blstm.scan(x, mask=mask, **kwargs)
        return x

    def fold(self, x, mask=None, **kwargs):
        for blstm in self.blstms[:-1]:
            x = blstm.scan(x, mask=mask, **kwargs)
        return blstms[-1].fold(x, mask=mask, **kwargs)

    def unfold(self, x, steps, **kwargs):
        x = blstms[0].unfold(x, steps, **kwargs)
        for blstm in self.blstms[1:]:
            x = blstm.scan(x, **kwargs)
        return x








