import cPickle as pickle
import numpy as np
import math

import compose
import solvers

def close_to_zero(x, tol=1e-6):
    return math.abs(x) <= tol

def null_func(*args, **kwargs):
    pass

class Network(object):

    def __init__(self, f, solver=solvers.NullSolver, dtype=np.float64):
        self.f = f
        self.W = f.weights(dtype=dtype)
        self.solver = solver

    def fit(self, train_data, validate_data=None, validate_every=1, save=null_func
            , *args, **kwargs):
        for info in self.solver(self.f.error_grad, self.W, train_data):
            if close_to_zero(info.iters % validate_every):
                if validate_data is not None:
                    info.err, info.extras = self.validate(validate_data, **kwargs)
                save(self, info)

    def validate(self, data, **kwargs):
        err = 0
        extras = {}
        n = 0
        for X,Y in data:
            for batch_err,batch_extras,batch_n in self.f.error(self.W, X, Y, **kwargs):
                n += batch_n
                err += batch_err
                for k,v in batch_extras.iteritems():
                    extras[k] = extras.get(k, 0) + v
        err = err/n if n != 0 and err != 0 else 0
        for k in extras:
            extras[k] = extras[k]/n if n != 0 and extras[k] != 0 else 0
        return err, extras

    def predict(self, data, **kwargs):
        for X in data:
            yield self.f(self.W, X, **kwargs)

    def __call__(self, data, **kwargs):
        return self.predict(data, **kwargs)

    def __getattr__(self, attr):
        call = self.f.__getattribute__(attr)
        if callable(call):
            def wrapped(data, *args, **kwargs):
                for X in data:
                    yield call(self.W, X, *args, **kwargs)
            return wrapped
        return call
                
    @property
    def dtype(self): return self.W.dtype

    @dtype.setter
    def dtype(self, t):
        if self.W.dtype != t:
            self.W = self.W.astype(t, copy=False)

    def save(self, f):
        pickle.dump(self, f)

    @staticmethod
    def load(f):
        return pickle.load(f)

    #do not pickle the solver states
    def __getstate__(self):
        x = {}
        x['f'] = self.f
        x['solver'] = self.solver
        x['W'] = self.W
        return x

    def __setstate__(self, st):
        if type(st) is tuple:
            self.f, self.solver, _, self.W = st
        else:
            self.f = st['f']
            self.solver = st['solver']
            self.W = st['W']

class Map(Network):

    def __init__(self, *args, **kwargs):
        super(Network, self).__init__(*args, **kwargs)

    def predict(self, X, bptt=None):
        bptt = bptt if bptt is not None else self.trunc_bptt
        n = X.shape[0]
        if bptt > 0:
            for i in xrange(0, n, bptt):
                yield self.f.forward(self.W, X[i:i+bptt], train=False)
                self.f.advance()
        else:
            yield self.f.forward(self.W, X, train=False)
        self.f.reset()

    def error(self, X, Y, bptt=None): #TODODODODODO
        pass

    def train(self, X, Y, bptt=None):
        bptt = bptt if bptt is not None else self.trunc_bptt
        if self.dW.shape != self.W.shape:
            self.dW.resize(self.W.shape)  
        n = X.shape[0]
        if bptt > 0:
            i = 0
            def grad(w, dw):
                yh = self.f.forward(w, X[i:i+bptt], train=True)
                dx = self.f.backward(w, Y[i:i+bptt], dw)
                return yh
            for i in xrange(0, n, bptt):
                yield self.solver.step(grad, self.W, self.dW)
                self.f.advance()
        else:
            def grad(w, dw):
                yh = self.f.forward(w, X, train=True)
                dx = self.f.backward(w, Y, dw)
                return yh
            yield self.solver.step(grad, self.W, self.dW)
        self.f.reset()

class Fold(Network):

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

    def predict(self, X, bptt=None):
        bptt = bptt if bptt is not None else self.trunc_bptt
        n = X.shape[0]
        if bptt > 0:
            for i in xrange(0, n, bptt):
                yh = self.f.forward(self.W, X[i:i+bptt], train=False)
                self.f.advance()
        else:
            yh = self.f.forward(self.W, X, train=False)
        yield yh
        self.f.reset()

    def error(self, X, Y, bptt=None): #TODODODO
        pass

    def train(self, X, Y, bptt=None):
        bptt = bptt if bptt is not None else self.trunc_bptt
        if self.dW.shape != self.W.shape:
            self.dW.resize(self.W.shape)
        n = X.shape[0]
        if bptt > 0:
            i = 0
            def grad(w, dw):
                dw[:] = 0
                yh = self.f.forward(w, X[i:i+bptt], train=True)
                dx = self.f.backward(w, Y, dw)
                return yh
            for i in xrange(0, n, bptt):
                yh = self.solver.step(grad, self.W, self.dW)
                self.f.advance()
        else:
            def grad(w, dw):
                dw[:] = 0
                yh = self.f.forward(w, X, train=True)
                dx = self.f.backward(w, Y, dw)
                return yh
            yh = self.solver.step(grad, self.W, self.dW)
        yield yh
        self.f.reset()
            
    
        
