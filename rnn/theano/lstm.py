import sys

import theano
import theano.tensor as th
import numpy as np

from activation import fast_tanh, fast_sigmoid
from batchnormalization import BatchNormalizer
sys.path.append("../../")
from rnn.initializers import orthogonal

def step(ifog, y0, c0, wy, iact=fast_sigmoid, fact=fast_sigmoid, oact=fast_sigmoid, gact=fast_tanh
        , cact=fast_tanh, mask=None, activation=lambda x: x):
    m = y0.shape[1]
    ifog = ifog + th.dot(y0, wy)
    i = iact(ifog[:,:m])
    f = fact(ifog[:,m:2*m])
    o = oact(ifog[:,2*m:3*m])
    g = gact(ifog[:,3*m:])
    c = c0*f + i*g
    y = activation(o*cact(c))
    if mask is not None:
        #mask = th.shape_padright(mask)
        I = (1-mask).nonzero()
        y = th.set_subtensor(y[I], y0[I])
        c = th.set_subtensor(c[I], c0[I])
        #y = y*mask + y0*(1-mask)
        #c = c*mask + c0*(1-mask)
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
                 , cact=fast_tanh, forget_bias=3, batch_norm=False):
        w = np.random.randn(1+ins+units, 4*units).astype(dtype)
        w[0] = 0
        w[0,units:2*units] = forget_bias
        init(w[1:])
        self.ws = theano.shared(w, name=name, borrow=True)
        self.c0 = theano.shared(np.zeros((units)),borrow=True)
        self.y0 = theano.shared(np.zeros((units)),borrow=True)
        #self.y0 = th.zeros((1,self.units))
        #self.c0 = th.zeros((1,self.units))
        self.name = name
        self.iact = iact
        self.fact = fact
        self.oact = oact
        self.gact = gact
        self.cact = cact

        if batch_norm:
            self.norm = BatchNormalizer(units)
        else:
            self.norm = None

    @property
    def weights(self): 
        if self.norm != None:
            return [self.ws] + self.norm.weights()
        else:
            return [self.ws]

    @property
    def ins(self): return self.ws.get_value(borrow=True).shape[0] - (1 + self.units)

    @property
    def units(self): return self.ws.get_value(borrow=True).shape[1]//4

    @property
    def l2(self):
        return th.sum(self.weights[0]**2)

    def delete(self, del_units, history=None):
        #print del_units
        weights = self.ws.get_value()
        unit_count = self.units
        for unit in del_units:
            for i in range(3,-1,-1):
                weights = np.delete(weights, i*unit_count + unit, 1)
            unit_count -= 1
            weights = np.delete(weights, unit + 1 + self.ins, 0)

        if history is not None:
            for lstm_hist in history:
                hist = lstm_hist.get_value()
                unit_count = self.units
                for unit in del_units:
                    for i in range(3,-1,-1):
                        hist = np.delete(hist, i*unit_count + unit, 1)
                    unit_count -= 1
                    hist = np.delete(hist, unit + 1 + self.ins, 0)
                lstm_hist.set_value(hist)

        self.ws.set_value(weights)
        self.y0.set_value(np.zeros((self.units)))
        self.c0.set_value(np.zeros((self.units)))

    def add(self, add_units, history=None):
        weights = self.ws.get_value()
        #print weights
        for i in range (4, 0, -1):
            random = np.random.rand(add_units, weights.shape[0])
            normalized_random = random/np.reshape(np.linalg.norm(random, axis=1), (-1,1))
            weights = np.insert(weights, i*self.units, normalized_random, axis = 1)
        random = np.random.rand(add_units, weights.shape[1])
        normalized_random = random/np.reshape(np.linalg.norm(random, axis=1), (-1,1))
        weights = np.insert(weights, self.units+1+self.ins, normalized_random, axis = 0)
        #print weights

        if history is not None:
            for lstm_hist in history:
                hist = lstm_hist.get_value()
                for i in range (4, 0, -1):
                    zeros = np.zeros((add_units, hist.shape[0]))
                    hist = np.insert(hist, i*self.units, zeros, axis = 1)
                zeros = np.zeros((add_units, hist.shape[1]))
                hist = np.insert(hist, self.units+1+self.ins, zeros, axis = 0)
                lstm_hist.set_value(hist)

        self.ws.set_value(weights)
        self.y0.set_value(np.zeros((self.units)))
        self.c0.set_value(np.zeros((self.units)))

    # Changes weights so it has the right number of inputs for 
    def delete_ins(self, del_ins, history=None):
        #print del_ins
        weights = self.ws.get_value()
        print weights.shape
        for ins in del_ins:
            weights = np.delete(weights, ins + 1, 0)
        print weights.shape
        self.ws.set_value(weights)
        print self.ins

        if history is not None:
            for lstm_hist in history:
                hist = lstm_hist[0].get_value()
                for ins in del_ins:
                    hist = np.delete(hist, ins + 1, 0)
                print hist.shape
                lstm_hist[0].set_value(hist)

    def add_ins(self, add_ins, history=None):
        #print del_ins
        weights = self.ws.get_value()
        for ins in add_ins:
            random = np.random.rand(ins[1], weights.shape[1])
            normalized_random = random/np.reshape(np.linalg.norm(random, axis=1), (-1,1))
            weights = np.insert(weights, ins[0] + 1, normalized_random, 0)
        self.ws.set_value(weights)

        if history is not None:
            for lstm_hist in history:
                hist = lstm_hist[0].get_value()
                for ins in add_ins:
                    random = np.random.rand(ins[1], hist.shape[1])
                    normalized_random = random/np.reshape(np.linalg.norm(random, axis=1), (-1,1))
                    hist = np.insert(hist, ins[0] + 1, normalized_random, 0)
                lstm_hist[0].set_value(hist)

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

    def build_init_states(self, x, y0, c0):
        if 'int' in x.dtype:
            shape = x.shape[1:]
        else:
            shape = x.shape[1:-1]
        z = th.shape_padright(th.zeros(shape))
        if y0 is None:
            y0 = self.y0
        if c0 is None:
            c0 = self.c0
        return z+y0, z+c0

    def scanl(self, x, y0=None, c0=None, mask=None, **kwargs):
        y0, c0 = self.build_init_states(x, y0, c0)
        y, c = scanl(self.ws, y0, c0, x, mask=mask, iact=self.iact, fact=self.fact, oact=self.oact
                     , gact=self.gact, cact=self.cact, **kwargs)
        if self.norm != None:
            y = self.norm.scan(y)
        return y, c

    def scanr(self, x, y0=None, c0=None, mask=None, **kwargs):
        y0, c0 = self.build_init_states(x, y0, c0)
        y0, c0 = scanr(self.ws, y0, c0, x, mask=mask, iact=self.iact, fact=self.fact, oact=self.oact
                     , gact=self.gact, cact=self.cact, **kwargs)
        if self.norm != None:
            y0 = self.norm.scan(y0)
        return y0, c0

    def foldl(self, x, y0=None, c0=None, mask=None, **kwargs):
        if y0 is None:
            #y0 = self.cact(self.y0)
            y0 = self.y0
        if c0 is None:
            c0 = self.c0
        return foldl(self.ws, y0, c0, x, mask=mask, iact=self.iact, fact=self.fact, oact=self.oact
                     , gact=self.gact, cact=self.cact, **kwargs)

    def foldr(self, x, y0=None, c0=None, mask=None, **kwargs):
        if y0 is None:
            #y0 = self.cact(self.y0)
            y0 = self.y0
        if c0 is None:
            c0 = self.c0
        return foldr(self.ws, y0, c0, x, mask=mask, iact=self.iact, fact=self.fact, oact=self.oact
                     , gact=self.gact, cact=self.cact, **kwargs)

    def unfoldl(self, x, steps, y0=None, c0=None, **kwargs):
        if y0 is None:
            #y0 = self.cact(self.y0)
            y0 = self.y0
        if c0 is None:
            c0 = self.c0
        return unfoldl(self.ws, y0, c0, x, steps, iact=self.iact, fact=self.fact, oact=self.oact
                , gact=self.gact, cact=self.cact, **kwargs)

    def unfoldr(self, x, steps, y0=None, c0=None, **kwargs):
        if y0 is None:
            #y0 = self.cact(self.y0)
            y0 = self.y0
        if c0 is None:
            c0 = self.c0
        return unfoldr(self.ws, y0, c0, x, steps, iact=self.iact, fact=self.fact, oact=self.oact
                , gact=self.gact, cact=self.cact, **kwargs)

class LayeredLSTM(object):
    def __init__(self, ins, layers, batch_norm=False,**kwargs):
        self.lstms = []
        for n in layers:
            self.lstms.append(LSTM(ins, n, batch_norm=batch_norm, **kwargs))
            ins = n

    @property
    def units(self):
        return sum(lstm.units for lstm in self.lstms)

    @property
    def weights(self):
        weights = []
        for lstm in self.lstms:
           weights += lstm.weights
        return weights

    @property
    def l2(self):
        return sum(lstm.l2 for lstm in self.lsmts)

    def split(self, vector):
        if vector is not None:
            splits = []
            i = 0
            for lstm in self.lstms:
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
        ys = th.concatenate(ys, axis=-1)
        cs = th.concatenate(cs, axis=-1)
        return y, ys, cs

class BLSTM(object):
    def __init__(self, ins, units, batch_norm=False, **kwargs):
        # ins = dimension for set of parameters
        self.nl = units // 2 + units % 2
        self.nr = units // 2
        self.lstml = LSTM(ins, self.nl, batch_norm=batch_norm, **kwargs)
        self.lstmr = LSTM(ins, self.nr, batch_norm=batch_norm, **kwargs)

    @property
    def weights(self): return self.lstml.weights + self.lstmr.weights

    @property
    def units(self): return self.nl + self.nr

    @property
    def l2(self):
        return self.lstml.l2 + self.lstmr.l2

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
    def __init__(self, ins, layers, batch_norm=False, **kwargs):
        self.blstms = []
        for n in layers:
            self.blstms.append(DiffBLSTM(ins, n, batch_norm=batch_norm, **kwargs))
            ins = n

    @property
    def weights(self):
        weights = []
        for blstm in self.blstms:
            weights += blstm.weights
        return weights

    @property
    def units(self): return sum(blstm.units for blstm in self.blstms)

    @property
    def l2(self):
        return sum(blstm.l2 for blstm in self.blstms)

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

class DiffLSTM(object):
    def __init__(self, ins, units, batch_norm = False, Td=0.05, k=5, Mmin=0.025, Padd=0.5, Pdel=0.5, **kwargs):
        self.lstm = LSTM(ins, units, **kwargs)
        self.Td = Td # Deletion Threshold - if falls below threshold then marked for possible deletion
        self.k = k # Number of units with multiplier below threshold (will adjust number of units if less/more than k)
        self.Mmin = Mmin # The minimum that a multiplier can be
        self.Padd = Padd # Probability that a unit to be added is added
        self.Pdel = Pdel # Probabiiity that a unit to be deleted is deleted
        self.multiplier = theano.shared(np.ones((units))*self.Td)
        if batch_norm:
            self.norm = BatchNormalizer(units)
        else:
            self.norm = None

    @property
    def units(self): return self.lstm.units

    @property
    def weights(self): 
        if self.norm != None:
            return self.lstm.weights + [self.multiplier] + self.norm.weights()
        else:
            return self.lstm.weights + [self.multiplier]

    @property
    def l2(self):
        return self.lstm.l2

    def update(self, history=None):
        #multip = np.absolute(self.multiplier.get_value())
        multip = self.multiplier.get_value()
        below_threshold = []
        #print "Multip before: %s" % multip
        for i in range(len(multip)):
            if abs(multip[i]) < self.Td:
                below_threshold += [i]
            if abs(multip[i]) < self.Mmin:
                if multip[i] < 0:
                    multip[i] = -self.Mmin
                else:
                    multip[i] = self.Mmin
        diff = self.k - len(below_threshold)
        del_units = []
        if diff < 0:
            below_threshold = np.array(below_threshold)
            np.random.shuffle(below_threshold)
            potential_remove = below_threshold[:-diff]
            # Preserve potentially removed unit with probability 1-Pdel
            for i in potential_remove:
            	if np.random.rand() < self.Pdel:
                    del_units.append(i)
            if del_units != []:
                del_units.sort()
                del_units = del_units[::-1]
                # Updates weights for batch normalizer
                if self.norm != None:
                    norm_hist = [hist[2:4] for hist in history]
                    history = [hist[0:2] for hist in history]
                    self.norm.delete(del_units, norm_hist)
                self.delete(del_units, history)
        add_units = 0
        if diff > 0:
            for i in range(diff):
                # Adds unit with probability Padd
                if np.random.rand() < self.Padd:
                    add_units += 1
            if add_units > 0:
                # Updates weights for batch normalizer
                if self.norm != None:
                    norm_hist = [hist[2:4] for hist in history]
                    history = [hist[0:2] for hist in history]
                    self.norm.add(add_units, norm_hist)
                self.add(add_units, history)
        #returns list of units to be deleted and number of units to be added to end (to inform next layer of unit change)
        print "Multiplier: %s" % multip
        if del_units != []:
            # deletions are made as a list of point deletions
            return [[], del_units]
        elif add_units != 0:
            # additions are made as a list of location of the additon and the number of units added
            return [[(self.units - add_units, add_units)], []]
        else:
            return [[],[]]

    # Updates LSTM if it is after a differentiable layer
    def update_ins(self, change, history=None):
        adds, dels = change
        print "adds = %s" % adds
        print "dels = %s" % dels
        if adds != []: 
        # Adding unit
            self.lstm.add_ins(adds, history=history)
        if dels != []: 
        # Removing unit
            self.lstm.delete_ins(dels, history=history)

    # Deletes selected units from multiplier and lstm layer
    def delete(self, del_units, history=None):
        multip = self.multiplier.get_value()
        for i in del_units:
            multip = np.delete(multip,i,0)
        self.multiplier.set_value(multip)
        lstm_history = None
        if history is not None:
            lstm_history = []
            for hist in history:
                hist_multip = hist[1].get_value()
                for i in del_units:
                    hist_multip = np.delete(hist_multip,i,0)
                hist[1].set_value(hist_multip)
                lstm_history += [hist[0]]
        self.lstm.delete(del_units, lstm_history)

    # Adds units to multiplier and lstm layer
    def add(self, add_units, history=None):
        multip = self.multiplier.get_value()
        multip = np.concatenate((multip, np.array([self.Td] * add_units)))
        self.multiplier.set_value(multip)
        lstm_history = None
        if history is not None:
            lstm_history = []
            for hist in history:
                hist_multip = hist[1].get_value()
                hist_multip = np.concatenate((hist_multip, np.array([self.Td] * add_units)))
                hist[1].set_value(hist_multip)
                lstm_history += [hist[0]]
        self.lstm.add(add_units, lstm_history)

    def scanl(self, x, y0=None, c0=None, mask=None, **kwargs):
        y, c = self.lstm.scanl(x, y0=None, c0=None,  mask = mask, **kwargs)
        y = y * self.multiplier
        if self.norm != None:
            y = self.norm.scan(y)
        return y, c

    def scanr(self, x, y0=None, c0=None, mask=None, **kwargs):
        y, c = self.lstm.scanr(x, y0=None, c0=None,  mask = mask, **kwargs)
        y = y * self.multiplier
        if self.norm != None:
            y = self.norm.scan(y)
        return y, c

class DiffBLSTM(object):
    def __init__(self, ins, units, batch_norm=False, **kwargs):
        # ins = dimension for set of parameters
        nl = units // 2 + units % 2 
        nr = units // 2
        self.lstml = DiffLSTM(ins, nl, batch_norm=batch_norm, **kwargs)
        self.lstmr = DiffLSTM(ins, nr, batch_norm=batch_norm, **kwargs)
        self.multiplier = th.concatenate([self.lstml.multiplier, self.lstmr.multiplier], axis = 0)
        self.bnorm = batch_norm

    @property
    def weights(self): return self.lstml.weights + self.lstmr.weights

    @property
    def units(self): return self.nl + self.nr

    @property
    def l2(self):
        return self.lstml.l2 + self.lstmr.l2

    @property
    def nl(self): return self.lstml.units

    @property
    def nr(self): return self.lstmr.units

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

    def update(self, history=None):
        if history != None:
            if self.bnorm:
                historyl = [hist[:4] for hist in history]
                historyr = [hist[4:] for hist in history]
            else:
                historyl = [hist[:2] for hist in history]
                historyr = [hist[2:] for hist in history]
            history = [historyl] + [historyr]
            updatel = self.lstml.update(history=history[0])
            updater = self.lstmr.update(history=history[1])
        # If both lstms are changed then need to modify the indices of the changes
        # index_change = self.nl
        # if (updater[0] != [] or updater[1] != []) and (updatel[0] != [] or updatel[1] != []):
        #     index_change -= len(updatel[1])
        #     if updatel[0] != []:
        #         index_change = updatel[0][0][1]
        #     if updater[0] != []:
        #         return [updatel[0]+[(updater[0][0][0], updater[0][0][1] + index_change)], updatel[1]]
        #     else:
        #         return [updatel[0], updatel[1] + [delr+index_change for delr in updater[1]]]
        # else:
        #     return [updater[0]+updatel[0], updater[1]+updatel[1]]
        unit_change = 0

        # Because the units of the left lstm changed after adding
        if updatel[0] != []:
            unit_change = -updatel[0][0][1]
        # Because the units of the left lstm changed after deleting
        if updatel[1] != []:
            unit_change = len(updatel[1])
        if updater[0] != []:
            updater[0][0] = (self.nl + updater[0][0][0] + unit_change, updater[0][0][1])
        if updater[1] != []:
            updater[1] = [delete + self.nl + unit_change for delete in updater[1]]
        return [updater[0] + updatel[0], updater[1] + updatel[1]]

    # Updates LSTM ins if it is after a differentiable layer
    def update_ins(self, change, history=None):
        if history != None:
            if self.bnorm:
                historyl = [hist[0:2] for hist in history]
                historyr = [hist[4:6] for hist in history]
            else:
                historyl = [hist[:2] for hist in history]
                historyr = [hist[2:] for hist in history]
            history = [historyl] + [historyr]
            self.lstml.update_ins(change, history=history[0])
            self.lstmr.update_ins(change, history=history[1])
        else:
            self.lstml.update_ins(change, history=history)
            self.lstmr.update_ins(change, history=history)

class DiffLayeredBLSTM(object):
    def __init__(self, ins, layers, batch_norm=False, **kwargs):
        self.blstms = []
        for n in layers:
            self.blstms.append(DiffBLSTM(ins, n, batch_norm=batch_norm, **kwargs))
            ins = n
        self.multiplier = []
        for n in self.blstms:
            self.multiplier += [n.multiplier]
        self.multiplier = th.concatenate(self.multiplier, axis = 0)
        self.bnorm = batch_norm

    @property
    def weights(self):
        weights = []
        for blstm in self.blstms:
            weights += blstm.weights
        return weights

    @property
    def units(self): return sum(blstm.units for blstm in self.blstms)

    @property
    def l2(self):
        return sum(blstm.l2 for blstm in self.blstms)

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

    def update(self, history=None):
        if history is not None:
            if self.bnorm:
                numw = 8 # number of weights for each layer of the BLSTM
            else:
                numw = 4
            for n in range(len(self.blstms)):
                his = [hist[numw*n:numw*(n+1)] for hist in history]
                changes = self.blstms[n].update(history=his)
                if n < len(self.blstms)-1:
                    his = [hist[numw*(n+1):numw*(n+2)] for hist in history]
                    self.blstms[n+1].update_ins(changes, history=his)
            return changes # Returns the changes of the last layer
        else:
            for n in range(len(self.blstms)):
                changes = self.blstm[n].update(history=history)
                if n < len(self.blstms)-1:
                    self.blstms[n+1].update_ins(changes, history=history)
            return changes # Returns the changes of the last layer

# LSTM whose units are separated into n equal sized sections.
# The output from each section is dotted with the others and added to the loss
# This should force the outputs of each section to be different as to capture independent aspects of the model
# If multiple sections fall below the threshold, they are deleted
# If there are not enough sections below the threshold, more are added
# No multiplier right now
class ParallelLSTM(DiffLSTM):
    def __init__(self, ins, units, sections, batch_norm=False, **kwargs):
        DiffLSTM.__init__(self, ins, units*sections, batch_norm, **kwargs)
        self.secunits = units

    @property
    def sections(self): return DiffLSTM.units.fget(self) / self.secunits

    @property
    def weights(self): return self.lstm.weights

    #def update(self, history=None):
        # Todo: make update function

    def scanl(self, x, y0=None, c0=None, mask=None, **kwargs):
        y, c = self.lstm.scanl(x, y0=None, c0=None,  mask = mask, **kwargs)
        if self.norm != None:
            y = self.norm.scan(y)
        return y, c

    def scanr(self, x, y0=None, c0=None, mask=None, **kwargs):
        y, c = self.lstm.scanr(x, y0=None, c0=None,  mask = mask, **kwargs)
        if self.norm != None:
            y = self.norm.scan(y)
        return y, c
