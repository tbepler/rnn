import ctypes
import numpy as np
import math

import algorithms as algo
import initializers as init

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
    def wshape(self): return (self.inputs+self.outputs+1, 4*self.outputs)

    @property
    def wsize(self): return (self.inputs+self.outputs+1)*4*self.outputs

    def states(self, dtype=np.float64):
        return np.zeros(6*self.outputs, dtype=dtype)

    @property
    def sshape(self): return 6*self.outputs

    @property
    def ssize(self): return 6*self.outputs
            
    def forward(self, X, Y, W, S, train=None):
        (k,b,_) = X.shape
        #self.advance()
        self.resize_fwd(k, b, W.dtype)
        algo.lstmfw(W, X, self.S, self._Y)
        self.X = X
        return self._Y[1:,:,:]

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
