import cPickle as pickle
import numpy as np

import compose
import solvers

class Network(object):

    def __init__(self, f, trunc_bptt=0, solver=solvers.NullSolver, dtype=np.float64):
        self.f = f
        self.trunc_bptt = trunc_bptt
        self.W = f.weights(dtype=dtype)
        self.dW = np.zeros(0, dtype=dtype)
        self.solver = solver
        self._dtype = dtype
        self.solver.dtype = dtype

    @property
    def dtype(self): return self._dtype

    @dtype.setter
    def dtype(self, t):
        if self._dtype != t:
            self._dtype = t
            self.solver.dtype = self.dtype
            self.W = self.W.astype(t, copy=False)
            self.dW = self.dW.astype(t, copy=False)

    def save(self, f):
        pickle.dump(self, f)

    @staticmethod
    def load(f):
        return pickle.load(f)

    #do not pickle the solver states
    def __getstate__(self):
        x = {}
        x['f'] = self.f
        x['bptt'] = self.trunc_bptt
        x['solver'] = self.solver
        x['dtype'] = self._dtype
        x['W'] = self.W
        return x

    def __setstate__(self, st):
        if type(st) is tuple:
            self.f, self.solver, self._dtype, self.W = st
            self.trunc_bptt = 0
        else:
            self.f = st['f']
            self.trunc_bptt = st.get('bptt', 0)
            self.solver = st['solver']
            self._dtype = st['dtype']
            self.W = st['W']
        self.dW = np.zeros(0, dtype=self.dtype)

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
            
    
        
