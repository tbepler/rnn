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
        y = y*mask + y0*(1-mask)
        c = c*mask + c0*(1-mask)
    return y, c

def split(w):
    m = w.shape[1]/4
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
                 , cact=fast_tanh, forget_bias=3):
        w = np.random.randn(1+units+ins, 4*units).astype(dtype)
        w[0] = 0
        w[0,units:2*units] = forget_bias
        init(w[1:])
        self.ws = theano.shared(w, name=name, borrow=True)
        #self.c0 = theano.shared(np.zeros(units,dtype=dtype),borrow=True)
        self.c0 = th.zeros((1,units))
        #self.y0 = theano.shared(np.zeros(units,dtype=dtype),borrow=True)
        self.y0 = th.zeros((1,units))
        self.name = name
        self.iact = iact
        self.fact = fact
        self.oact = oact
        self.gact = gact
        self.cact = cact

    @property
    def weights(self): return [self.ws]

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
        self.c0 = th.zeros((1,state['weights'].shape[1]//4))
        #self.y0 = theano.shared(state['y0'], borrow=True)
        self.y0 = th.zeros((1,state['weights'].shape[1]//4))
        acts = state['activations']
        self.iact = acts[0]
        self.fact = acts[1]
        self.oact = acts[2]
        self.gact = acts[3]
        self.cact = acts[4]

    def scanl(self, x, y0=None, c0=None, mask=None, **kwargs):
        if y0 is None:
            #y0 = self.cact(self.y0)
            y0 = self.y0
        if c0 is None:
            c0 = self.c0
        return scanl(self.weights, y0, c0, x, mask=mask, iact=self.iact, fact=self.fact, oact=self.oact
                     , gact=self.gact, cact=self.cact, **kwargs)

    def scanr(self, x, y0=None, c0=None, mask=None, **kwargs):
        if y0 is None:
            #y0 = self.cact(self.y0)
            y0 = self.y0
        if c0 is None:
            c0 = self.c0
        return scanr(self.weights, y0, c0, x, mask=mask, iact=self.iact, fact=self.fact, oact=self.oact
                     , gact=self.gact, cact=self.cact, **kwargs)

    def foldl(self, x, y0=None, c0=None, mask=None, **kwargs):
        if y0 is None:
            #y0 = self.cact(self.y0)
            y0 = self.y0
        if c0 is None:
            c0 = self.c0
        return foldl(self.weights, y0, c0, x, mask=mask, iact=self.iact, fact=self.fact, oact=self.oact
                     , gact=self.gact, cact=self.cact, **kwargs)

    def foldr(self, x, y0=None, c0=None, mask=None, **kwargs):
        if y0 is None:
            #y0 = self.cact(self.y0)
            y0 = self.y0
        if c0 is None:
            c0 = self.c0
        return foldr(self.weights, y0, c0, x, mask=mask, iact=self.iact, fact=self.fact, oact=self.oact
                     , gact=self.gact, cact=self.cact, **kwargs)

    def unfoldl(self, x, steps, y0=None, c0=None, **kwargs):
        if y0 is None:
            #y0 = self.cact(self.y0)
            y0 = self.y0
        if c0 is None:
            c0 = self.c0
        return unfoldl(self.weights, y0, c0, x, steps, iact=self.iact, fact=self.fact, oact=self.oact
                , gact=self.gact, cact=self.cact, **kwargs)

    def unfoldr(self, x, steps, y0=None, c0=None, **kwargs):
        if y0 is None:
            #y0 = self.cact(self.y0)
            y0 = self.y0
        if c0 is None:
            c0 = self.c0
        return unfoldr(self.weights, y0, c0, x, steps, iact=self.iact, fact=self.fact, oact=self.oact
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

    def split(self, vector):
        if vector is not None:
            splits = []
            i = 0
            for lstm in lstms:
                n = lstm.units
                splits.append(vector[:,i:i+n])
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
        ys = T.concatenate(ys, axis=-1)
        cs = T.concatenate(cs, axis=-1)
        return y, ys, cs

class BLSTM(object):
    def __init__(self, ins, units, **kwargs):
        self.nl = units // 2 + units % 2
        self.nr = units // 2
        self.lstml = LSTM(ins, nl, **kwargs)
        self.lstmr = LSTM(ins, nr, **kwargs)

    @property
    def weights(self): return self.lstml.weights + self.lsmtr.weights

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
    def weights(self): return sum(blstm.weights for blstm in self.blstms)

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








