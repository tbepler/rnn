import theano
import theano.tensor as T
import numpy as np

class Momentum(object):
    def __init__(self, momentum=0.9, dtype=theano.config.floatX):
        self.momentum = theano.shared(np.cast[dtype](momentum))

    def setup(self, thetas, *args, **kwargs):
        self.shared = [theano.shared(np.zeros_like(theta.get_value())) for theta in thetas] 

    def __call__(self, deltas):
        updates = []
        new_deltas = []
        for velocity, delta in zip(self.shared, deltas):
            velocity_new = self.momentum*velocity + delta # (1-self.momentum)*delta
            updates.append((velocity, velocity_new))
            delta = (1-self.momentum)*velocity_new # scale update magnitude to prevent it from increasing to 1/(1-momentum)
            new_deltas.append(delta)
        return new_deltas, updates

class Nesterov(Momentum):
    def __call__(self, deltas):
        updates = []
        new_deltas = []
        for velocity, delta in zip(self.shared, deltas):
            #step = self.momentum*(self.momentum*velocity - (1-self.momentum)*delta) - (1-self.momentum)*delta
            v_new = self.momentum*velocity + delta
            step = self.momentum*v_new + delta
            updates.append((velocity, v_new))
            step *= (1-self.momentum) #rescale the magnitude
            new_deltas.append(step)
        return new_deltas, updates

class RMSprop(object):
    def __init__(self, rho=0.95, eps=1e-8, dtype=theano.config.floatX):
        self.rho = np.cast[dtype](rho)
        self.eps = np.cast[dtype](eps)
        self.dtype = dtype

    def setup(self, thetas, *args, **kwargs):
        if len(thetas) > 0:
            self.rho_shared = theano.shared(np.cast[self.dtype](0))
            self.shared = [theano.shared(np.zeros_like(theta.get_value())) for theta in thetas]

    def __call__(self, grads):
        rho = self.rho_shared
        updates = [(self.rho_shared, T.constant(self.rho))]
        conditioners = []
        for accum, g in zip(self.shared, grads):
            accum_new = rho*accum + (1-rho)*g**2
            updates.append((accum, accum_new))
            cond = 1/(T.sqrt(accum_new) + self.eps)
            conditioners.append(cond)
        return conditioners, updates

class SGD(object):
    def __init__(self, nu=0.01, decay=0, C=None, M=None, dtype=theano.config.floatX):
        self.C = C
        self.M = M
        self.nu = theano.shared(np.cast[dtype](nu))
        self.decay = theano.shared(np.cast[dtype](decay))
        self.iterations = theano.shared(np.cast[dtype](0))

    @property
    def lr(self):
        return self.nu*(1/(1+self.decay*self.iterations))
    
    def setup(self, thetas, *args, **kwargs):
        if self.C is not None:
            self.C.setup(thetas, *args, **kwargs)
        if self.M is not None:
            self.M.setup(thetas, *args, **kwargs)

    def updates(self, thetas, grads, *args, **kwargs):
        self.setup(thetas, *args, **kwargs)
        steps = grads
        updates = [(self.iterations, self.iterations+1)]
        if self.C is not None:
            cs, upd = self.C(grads)
            updates.extend(upd)
            steps = [c*g for c,g in zip(cs, grads)]
        if self.M is not None:
            new_steps, upd = self.M(steps)
            updates.extend(upd)
            steps = new_steps
        nu = self.lr 
        steps = [nu*step for step in steps]
        for theta, step in zip(thetas, steps):
            updates.append((theta, theta-step))
        return updates







