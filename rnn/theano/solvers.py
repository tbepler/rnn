import theano
import theano.tensor as th
import numpy as np

class NoDecay(object):
    def __call__(self, lr):
        return lr

class GeomDecay(object):
    def __init__(self, rate):
        self.rate = rate
    def __call__(self, lr):
        return lr*self.rate

class SGD(object):
    def __init__(self, lr, mom=0.9, decay=NoDecay()):
        self.lr = lr
        self.mom = mom
        self.decay = decay

    def compile(self, weights, inputs, err, extras):
        gws = th.grad(err, weights)
        vs = [theano.shared(w.get_value()*0, broadcastable=w.broadcastable) for w in weights]
        lr = th.scalar()
        #mom = th.constant(self.mom)
        updates = []
        for w, gw, v in zip(weights, gws, vs):
            vnext = self.mom*v + (1-self.mom)*gw
            #vnext = v + gw
            updates.append((v, vnext))
            updates.append((w, w-lr*vnext))
        return theano.function([lr]+inputs, [err]+extras, updates=updates)
        
    def __call__(self, data, weights, inputs, err, extras, max_iters=-1):
        f = self.compile(weights, inputs, err, extras)
        n = len(data)
        i = 0
        while i < max_iters or max_iters < 0:
            j = 0.0
            for args in data:
                ret = f(self.lr, *args)
                j += 1
                yield i+j/n, ret
            i += 1
            self.lr = self.decay(self.lr)

class RMSprop(object):
    def __init__(self, lr, rho=0.95, eps=1e-5, mom=0.9, decay=NoDecay()):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.mom = mom
        self.decay = decay

    def compile(self, weights, inputs, err, extras):
        gws = th.grad(err, weightss)
        ms = [theano.shared(w.get_value()*0, broadcastable=w.broadcastable) for w in weights]
        vs = [theano.shared(w.get_value()*0, broadcastable=w.broadcastable) for w in weights]
        lr = th.scalar()
        rho = self.rho
        eps = self.eps
        mom = self.mom
        updates = []
        for w, gw, m, v in zip(ws, gws, ms, vs):
            mnext = rho*m + (1-rho)*gw*gw
            vnext = mom*v + (1-mom)*gw/(th.sqrt(mnext)+eps)
            updates.extend([(m,mnext), (v,vnext), (w, w-lr*vnext)])
        return theano.function([lr]+inputs, [err]+extras, updates=updates)

    def __call__(self, data, weights, inputs, err, extras, max_iters=-1):
        f = self.compile(weights, inputs, err, extras)
        n = len(data)
        i = 0
        while i < max_iters or max_iters < 0:
            j = 0.0
            for args in data:
                ret = f(self.lr, *args)
                j += 1
                yield i+j/n, ret
            i += 1
            self.lr = self.decay(self.lr)

            
