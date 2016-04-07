import ctypes
import numpy as np
import math

import rnn.kernel as algo
import rnn.initializers as init

class LSTM(object):

    def __init__(self, input_size, output_size, forget_bias=3, tau=float('inf')
                 , initializer = init.xavier):
        self.inputs = input_size
        self.outputs = output_size
        self.forget_bias = forget_bias
        self.tau = tau
        self.initializer = initializer

    def weights(self, dtype=np.float64):
        #print dir(self)
        W = np.zeros((self.inputs+self.outputs+1, 4*self.outputs), dtype=dtype)
        self.initializer(W[1:])
        W[0,self.outputs:2*self.outputs] = self.forget_bias
        return W
        
    @property
    def shape(self): return (self.inputs+self.outputs+1, 4*self.outputs)

    @property
    def size(self): return (self.inputs+self.outputs+1)*4*self.outputs
                         
    def forward(self, X, Y, W, S, train=None):
        (k,b,_) = X.shape
        #self.advance()
        self.resize_fwd(k, b, W.dtype)
        algo.lstmfw(W, X, self.S, self._Y)
        self.X = X
        return self._Y[1:,:,:]

    def alloc_states(self, X):
        (k,b,_) = X.shape
        return np.zeros((k+1,b,6*self.outputs), dtype=X.dtype)

    def alloc_outputs(self, X):
        (k,b,_) = X.shape
        return np.zeros((k+1,b,self.outputs), dtype=X.dtype)

    def fw_iter(self, W, X):
        S = None
        Yprev = None
        for x in X:
            if S is None:
                S = self.alloc_states(x)
            Y = self.alloc_outputs(x)
            if Yprev is not None:
                Y[0] = Yprev[-1]
            algo.lstmfw(W, x, S, Y)
            yield Y[1:]
            S[0] = S[-1]
            Yprev = Y
    
    def __call__(self, W, X, gradient=False):
        if gradient:
            X = list(X)
            S = [self.alloc_states(x) for x in X]
            Y = [self.alloc_outputs(x) for x in X]
            for i in xrange(len(X)):
                if i > 0:
                    S[i][0] = S[i-1][-1]
                    Y[i][0] = Y[i-1][-1]
                algo.lstmfw(W, X[i], S[i], Y[i])
            g = lambda dw, dy: self.grad(W, X, Y, S, dw, dy)
            return (y[1:] for y in Y), g
        return self.fw_iter(W, X)

    def grad(self, W, X, Y, S, dW, dY):
        dY = list(dY)
        dX = [np.zeros_like(x) for x in X]
        dS = None
        for i in reversed(xrange(len(X))):
            if dS is None:
                dS = np.zeros_like(S[i])
            algo.lstmbw(self.tau, W, X[i], S[i], Y[i], dY[i], dW, dX[i], dS)
            dS[-1] = dS[0]
        return dX

    def backward(self, W, dY, dW):
        (k,b,_) = dY.shape
        self.resize_bwd(k, b, W.dtype)
        #dW[:] = 0
        algo.lstmbw(self.tau, W, self.X, self.S, self._Y, dY, dW, self.dX, self.dS)
        return self.dX

if __name__ == '__main__':
    import test
    ins = 2
    outs = 2
    layer = LSTM(ins,outs)
    test.layer(layer, float_t=np.float64)
    test.layer(layer, float_t=np.float32)
