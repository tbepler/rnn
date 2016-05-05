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

class Momentum(object):
    def __init__(self, momentum, weights):
        self.momentum = momentum
        if momentum > 0:
            self.velocity = [theano.shared(w.get_value()*0) for w in weights]
        else:
            self.velocity = []

    def __call__(self, deltas):
        if self.momentum > 0:
            vel_upd = [self.momentum*v + (1-self.momentum)*d for v,d in zip(self.velocity,deltas)]
            return vel_upd, zip(self.velocity, vel_upd)
        return deltas, []

class SGD(object):
    def __init__(self, learning_rate, momentum=0.9, decay=NoDecay()):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay

    def _unscaled_deltas(self, weights, grads):
        return grads, []

    def _updates(self, weights, grads, learning_rate):
        delta, updates = self._unscaled_deltas(weights, grads)
        mom = Momentum(self.momentum, weights)
        mom_delta, mom_updates = mom(delta)
        w_upd = [(w, w-learning_rate*md) for w,md in zip(weights, mom_delta)]
        return updates + mom_updates + w_upd

    def _compile(self, inputs, outputs, weights, grads):
        learning_rate = th.scalar()
        updates = self._updates(weights, grads, learning_rate)
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
    
    def _unscaled_deltas(self, weights, grads):
        history = [theano.shared(w.get_value()*0) for w in weights]
        hist_upd = [self.rho*h + (1-self.rho)*g*g for h,g in zip(history, grads)]
        delta = [g/(th.sqrt(h)+self.eps) for g,h in zip(grads, hist_upd)]
        return delta, zip(history, hist_upd)

            
