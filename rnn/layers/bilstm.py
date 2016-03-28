import ctypes
import numpy as np
import math

import rnn.kernel as algo
import rnn.initializers as init

class BiLSTM(object):

    def __init__(self, input_size, output_size, forget_bias=3, tau=float('inf')
                 , initializer = init.xavier):
        self.inputs = input_size
        self._outputs = output_size
        self.forget_bias = forget_bias
        self.tau = tau
        self.initializer = initializer

    def __getstate__(self):
        x = {}
        x['inputs'] = self.inputs
        x['outputs'] = self._outputs
        x['forget_bias'] = self.forget_bias
        x['tau'] = self.tau
        x['init'] = self.initializer
        return x

    def __setstate__(self, x):
        if type(x) is tuple: #for backwards compatability
            if len(x) == 6:
                (self.inputs, self._outputs, self.forget_bias, self.tau
                 , self.initializer, _) = x
            elif len(x) == 5:
                (self.inputs, self._outputs, self.forget_bias, self.tau
                 , self.initializer) = x
            elif len(x) == 4:
                (self.inputs, self._outputs, self.forget_bias, self.tau) = x
                self.initializer = init.xavier
        else:
             self.inputs = x.get('inputs', 0)
             self._outputs = x.get('outputs', 0)
             self.forget_bias = x.get('forget_bias', 3)
             self.tau = x.get('tau', float('inf'))
             self.initializer = x.get('init', init.xavier) 

    def weights(self, dtype=np.float64):
        #print dir(self)
        W = np.zeros((2*(self.inputs+self._outputs+1), 4*self._outputs), dtype=dtype)
        self.initializer(W[1:self.inputs+self._outputs+1])
        W[0,self._outputs:2*self._outputs] = self.forget_bias
        self.initializer(W[self.inputs+self._outputs+2:])
        W[self.inputs+self._outputs+1,self._outputs:2*self._outputs] = self.forget_bias
        return W
        
    @property
    def shape(self): return (2*(self.inputs+self._outputs+1), 4*self._outputs)

    @property
    def size(self): return 2*(self.inputs+self._outputs+1)*4*self._outputs

    @property
    def outputs(self): return 2*self._outputs

    def alloc_states(self, X):
        (k,b,_) = X.shape
        return np.zeros((k+1,b,6*self._outputs), dtype=X.dtype)

    def alloc_outputs(self, X):
        (k,b,_) = X.shape
        return np.zeros((k+2,b,2*self._outputs), dtype=X.dtype)

    def grad(self, W, X, Y, Sl, Sr, dW, dY):
        dY = list(dY)
        dX = [np.zeros_like(x) for x in X]
        dS = None
        #gradient of left fold
        Wl = W[0:self._outputs+self.inputs+1]
        dWl = dW[0:self._outputs+self.inputs+1]
        for i in reversed(xrange(len(X))):
            if dS is None:
                dS = np.zeros_like(Sl[i])
            algo.lstmbw(self.tau, Wl, X[i], Sl[i], Y[i][0:-1,:,:self._outputs]
                        , dY[i][:,:,:self._outputs], dWl, dX[i], dS)
            dS[-1] = dS[0]
        #gradient of right fold
        Wr = W[self._outputs+self.inputs+1:]
        dWr = dW[self._outputs+self.inputs+1:]
        if dS is not None: dS[:] = 0
        for i in xrange(len(X)):
            algo.lstmrbw(self.tau, Wr, X[i], Sr[i], Y[i][1:,:,self._outputs:]
                         , dY[i][:,:,self._outputs:], dWr, dX[i], dS)
            dS[0] = dS[-1]
        return dX
    
    def __call__(self, W, X, gradient=False):
        X = list(X) #store chunks in list
        Sl = [self.alloc_states(x) for x in X]
        Sr = [self.alloc_states(x) for x in X]
        Y = [self.alloc_outputs(x) for x in X]
        #compute left fold
        Wl = W[0:self._outputs+self.inputs+1]
        for i in xrange(len(X)):
            if i > 0:
                Sl[i][0] = Sl[i-1][-1]
                Y[i][0,:,:self._outputs] = Y[i-1][-2,:,:self._outputs]
            algo.lstmfw(Wl, X[i], Sl[i], Y[i][0:-1,:,:self._outputs])
        #compute right fold
        Wr = W[self._outputs+self.inputs+1:]
        for i in reversed(xrange(len(X))):
            if i < len(X)-1:
                Sr[i][-1] = Sr[i+1][0]
                Y[i][-1,:,self._outputs:] = Y[i+1][1,:,self._outputs:]
            algo.lstmrfw(Wr, X[i], Sr[i], Y[i][1:,:,self._outputs:])

        iter_y = (y[1:-1] for y in Y)
        if gradient:
            g = lambda dw, dy: self.grad(W, X, Y, Sl, Sr, dw, dy)
            return iter_y, g
        return iter_y

if __name__ == '__main__':
    import test
    ins = 2
    outs = 2
    layer = BiLSTM(ins,outs)
    test.layer(layer, float_t=np.float64)
    test.layer(layer, float_t=np.float32)
