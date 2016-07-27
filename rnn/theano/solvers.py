import theano
import theano.tensor as th
import numpy as np
import lstm
from theano.compile.nanguardmode import NanGuardMode

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
        self.velocity = mom.velocity
        mom_delta, mom_updates = mom(delta)
        w_upd = [(w, w-learning_rate*md) for w,md in zip(weights, mom_delta)]
        return updates + mom_updates + w_upd

    def _compile(self, inputs, outputs, weights, grads):
        learning_rate = th.scalar()
        updates = self._updates(weights, grads, learning_rate)
        print learning_rate
        print inputs
        print outputs
        print updates
        return theano.function([learning_rate]+inputs, outputs, updates=updates, allow_input_downcast=True)

    def __call__(self, data, inputs, outputs, weights, grads=None, max_iters=-1):
        if grads is None:
            grads = theano.grad(outputs[0], weights)
        layers = inputs[-2]
        layerloc = inputs[-1]
        inputs = inputs[:-2]
        f = self._compile(inputs, outputs, weights, grads)
        n = len(data)
        i = 0
        while i < max_iters or max_iters < 0:
            j = 0.0
            for args in data:
                #print args
                ret = f(self.learning_rate, *args)
                j += 1
                yield i+j/n, ret
            i += 1
            self.learning_rate = self.decay(self.learning_rate)
            # Layer 0 updates and updates the historical data 
            # where change[0] are the additions and change[1] are the deletions from the outputs of layer 0
            if type(layers[0]) == lstm.DiffLSTM or type(layers[0]) == lstm.DiffBLSTM or type(layers[0]) == lstm.DiffLayeredBLSTM:
                change = layers[0].update(history=[self.history[layerloc[0][0]:layerloc[0][1]], self.velocity[layerloc[0][0]:layerloc[0][1]]])
                print change
                if change[0] != [] or change[1] != []:
                    # Layer 0's updates are passed into Layer 1 along with shared variables that need to be updated for the solver (history)
                    layers[1].update(change, history=[self.history[layerloc[1][0]:layerloc[1][1]], self.velocity[layerloc[1][0]:layerloc[1][1]]])

class RMSprop(SGD):
    def __init__(self, lr, rho=0.95, eps=1e-5, momentum=0.9, decay=NoDecay()):
        super(RMSprop, self).__init__(lr, momentum=momentum, decay=decay)
        self.rho = rho
        self.eps = eps
        self.history = []
    
    def _unscaled_deltas(self, weights, grads):
        print [w.name for w in weights]
        self.history = [theano.shared(w.get_value()*0) for w in weights]
        hist_upd = [self.rho*h + (1-self.rho)*g*g for h,g in zip(self.history, grads)]
        delta = [g/(th.sqrt(h)+self.eps) for g,h in zip(grads, hist_upd)]
        return delta, zip(self.history, hist_upd)
