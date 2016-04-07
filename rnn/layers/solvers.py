import ctypes
import numpy as np

import rnn.kernel as algo

class NullSolver(object):
    def step(self, W, dW):
        pass

class OLBFGS(object):

    def __init__(self, n=10, nu=0.1, lmbda=0, eps=1e-10, dtype=np.float64):
        self.n = n
        self.t = 0
        self.nu = nu
        self.lmbda = lmbda
        self.eps = eps
        self._dtype = dtype
        self.S = None
        self.Y = None

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, t):
        self._dtype = t
        if not self.S is None:
            self.S = self.S.astype(t, copy=False)
        if not self.Y is None:
            self.Y = self.Y.astype(t, copyt=False)    
        
    def step(self, f, w, dw):
        if self.S is None:
            self.S = np.zeros((self.n+1,w.size), self.dtype)
        if self.Y is None:
            self.Y = np.zeros((self.n+1,w.size), self.dtype)
        yh = algo.olbfgs(self.nu, self.lmbda, self.eps, self.t, f, w, dw, self.S, self.Y)
        self.t += 1
        return yh

    def __getstate__(self):
        return self.n, self.nu, self.lmbda, self.eps, self._dtype

    def __setstate__(self, x):
        self.n, self.nu, self.lmbda, self.eps, self._dtype = x
        self.t = 0
        self.S = None
        self.Y = None
        
    
class RMSProp(object):

    def __init__(self, nu=0.1, decay=1.0, rho=0.95, eps=1e-6, dtype=np.float64):
        self.nu = nu
        self.decay = decay
        self.rho = rho
        self.eps = eps
        self._dtype = dtype
        self.S = None

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, t):
        self._dtype = t
        if not self.S is None:
            self.S = self.S.astype(t, copy=False)
        
    def step(self, f, w, dw):
        if self.S is None:
            self.S = np.zeros(dw.size, self.dtype)
        else:
            assert self.S.size == dw.size
        y = f(w, dw)
        algo.rmsprop(self.nu, self.rho, self.eps, dw, w, self.S)
        self.nu *= self.decay
        return y

    def __getstate__(self):
        return self.nu, self.decay, self.rho, self.eps, self._dtype

    def __setstate__(self, x):
        self.nu, self.decay, self.rho, self.eps, self._dtype = x
        self.S = None
    
class Adadelta(object):

    def __init__(self, rho=0.95, eps=1e-6, dtype=np.float64):
        self.rho = rho
        self.eps = eps
        self._dtype = dtype
        self.S = None

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, t):
        self._dtype = t
        if not self.S is None:
            self.S = self.S.astype(t, copy=False)
        
    def step(self, f, w, dw):
        if self.S is None:
            self.S = np.zeros(2*w.size, self.dtype)
        else:
            assert self.S.size == 2*w.size
        y = f(w, dw)
        #adadelta :: rho, eps, n, grad, x, exp_grad, exp_delta
        algo.adadelta(self.rho, self.eps, dw, w, self.S[:w.size], self.S[w.size:])
        return y

    def __getstate__(self):
        return self.rho, self.eps, self._dtype

    def __setstate__(self, x):
        self.rho = x[0]
        self.eps = x[1]
        self._dtype = x[2]
        self.S = None
