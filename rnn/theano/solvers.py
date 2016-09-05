import theano
import theano.tensor as th
import numpy as np
import copy

class NoDecay(object):
    def __call__(self, lr, **kwargs):
        return lr

class GeomDecay(object):
    def __init__(self, rate):
        self.rate = rate
    def __call__(self, lr, iters=1):
        return lr*self.rate**iters

class Annealing(object):
    def __init__(self, T):
        self.T = T
    def __call__(self, lr, iters=1):
        return lr/(1+iters/self.T)

class Momentum(object):
    def __init__(self, momentum, weights):
        self.momentum = momentum
        if momentum > 0:
            self.velocity = [theano.shared(w.get_value()*0) for w in weights]
        else:
            self.velocity = []

    def __getstate__(self):
        return (self.momentum, [v.get_value() for v in self.velocity])

    def __setstate__(self, s):
        self.momentum, vs = s
        self.velocity = [theano.shared(v) for v in vs]

    def __call__(self, deltas):
        if self.momentum > 0:
            #mom_m1 = np.array(1-self.momentum, dtype=self.velocity[0].dtype)
            vel_upd = [self.momentum*v + (1-self.momentum)*d for v,d in zip(self.velocity,deltas)]
            return vel_upd, list(zip(self.velocity, vel_upd))
        return deltas, []

class SGD(object):
    def __init__(self, learning_rate, momentum=0.9, decay=NoDecay()):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.iters = theano.shared(0)

    def setup(self, weights):
        self.mom = Momentum(self.momentum, weights)

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        d['iters'] = self.iters.get_value()
        return d

    def __setstate__(self, s):
        self.__dict__.update(s)
        self.iters = theano.shared(self.iters)

    def _unscaled_deltas(self, weights, grads):
        return grads, []

    def _updates(self, weights, grads, learning_rate):
        delta, updates = self._unscaled_deltas(weights, grads)
        mom_delta, mom_updates = self.mom(delta)
        nu = self.decay(learning_rate, iters=self.iters)
        w_upd = [(w, w-nu*md) for w,md in zip(weights, mom_delta)]
        return updates + mom_updates + w_upd + [(self.iters, self.iters+1)]

    def _compile(self, inputs, outputs, weights, grads):
        learning_rate = th.scalar()
        updates = self._updates(weights, grads, learning_rate)
        #from theano.compile.nanguardmode import NanGuardMode
        #mode = NanGuardMode(nan_is_error=True, big_is_error=False, inf_is_error=False)
        #return theano.function([learning_rate]+inputs, outputs, updates=updates, mode=mode)
        return theano.function([learning_rate]+inputs, outputs, updates=updates)

    def __call__(self, data, inputs, outputs, weights, grads=None, max_iters=-1):
        if grads is None:
            grads = theano.grad(outputs[0], weights)
        f = self._compile(inputs, outputs, weights, grads)
        n = len(data)
        i = 0
        while i < max_iters or max_iters < 0:
            j = 0.0
            for args in data:
                ret = f(self.learning_rate, *args)
                j += 1
                yield i+j/n, ret
            i += 1
            self.learning_rate = self.decay(self.learning_rate)

class RMSprop(SGD):
    def __init__(self, lr, rho=0.95, eps=1e-5, momentum=0.9, decay=NoDecay()):
        super(RMSprop, self).__init__(lr, momentum=momentum, decay=decay)
        self.rho = rho
        self.eps = eps

    def setup(self, weights):
        super(RMSprop, self).setup(weights)
        self.rho_shared = theano.shared(np.cast[theano.config.floatX](0))
        self.history = [theano.shared(w.get_value()*0) for w in weights]

    def __getstate__(self):
        d = super(RMSprop, self).__getstate__()
        if 'rho_shared' in d:
            d['rho_shared'] = self.rho_shared.get_value()
        if 'history' in d:
            d['history'] = [h.get_value() for h in self.history]
        return d

    def __setstate__(self, d):
        super(RMSprop, self).__setstate__(d)
        if hasattr(self, 'rho_shared'):
            self.rho_shared = theano.shared(self.rho_shared)
        if hasattr(self, 'history'):
            self.history = [theano.shared(h) for h in self.history]
    
    def _unscaled_deltas(self, weights, grads):
        #rho = theano.shared(np.cast[theano.config.floatX](0))
        rho = self.rho_shared
        #history = [theano.shared(w.get_value()*0) for w in weights]
        history = self.history
        hist_upd = [rho*h + (1-rho)*g*g for h,g in zip(history, grads)]
        delta = [g/(th.sqrt(h)+self.eps) for g,h in zip(grads, hist_upd)]
        return delta, list(zip(history, hist_upd))+[(rho, th.constant(self.rho))]

            







