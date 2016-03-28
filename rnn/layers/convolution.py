import ctypes
import numpy as np
import math

import rnn.kernel as algo
import rnn.initializers as init

class Convolution(object):

    def __init__(self, input_size, output_size, window_size, initializer=init.xavier):
        self.inputs = input_size
        self.outputs = output_size
        self.window = window_size
        self.initializer = initializer
        self.Y = None
        self.Yprev = None
        self.dX = None

    def weights(self, dtype=np.float64):
        W = np.zeros((self.inputs*self.window, self.outputs), dtype=dtype)
        for i in xrange(self.window):
            self.initializer(W[i*self.inputs:(i+1)*self.inputs])
        W /= self.window
        return W
        
    @property
    def shape(self):
        return (self.inputs*self.window, self.outputs)

    @property
    def size(self):
        return self.inputs*self.window*self.outputs

    def __getstate__(self):
        return self.inputs, self.outputs, self.window, self.initializer

    def __setstate__(self, x):
        self.inputs, self.outputs, self.window, self.initializer = x
        self.Y = None
        self.Yprev = None
        self.dX = None

    def reset(self):
        self.Y[:] = 0
        self.Yprev[:] = 0

    def advance(self):
        k = self.Y.shape[0] - self.window + 1
        self.Yprev[:] = self.Y[k:]
            
    def resize_fw(self, k, b, dtype):
        if self.Y is None:
            self.Y = np.zeros((k+self.window-1,b,self.outputs), dtype=dtype)
        if self.Y.shape != (k+self.window-1,b,self.outputs):
            self.Y.resize((k+self.window-1,b,self.outputs), refcheck=False)
        if self.Y.dtype != dtype:
            self.Y = self.Y.astype(dtype, copy=False)
        if self.Yprev is None:
            self.Yprev = np.zeros((self.window-1,b,self.outputs), dtype=self.Y.dtype)
        if self.Yprev.shape != (self.window-1,b,self.outputs):
            self.Yprev.resize((self.window-1,b,self.outputs))
        if self.Yprev.dtype != self.Y.dtype:
            self.Yprev = self.Yprev.astype(dtype, copy=False)
            
    def forward(self, W, X, train=None):
        (k,b,_) = X.shape
        self.resize_fw(k, b, W.dtype)
        self.Y[:self.window-1] = self.Yprev
        self.Y[self.window-1:] = 0
        algo.convfw(W, X, self.Y)
        self.X = X
        #print self.Y
        return self.Y[:k]

    def resize_bw(self, k, b, dtype):
        if self.dX is None:
            self.dX = np.zeros((k,b,self.inputs), dtype=dtype)
        else:
            self.dX.resize((k,b,self.inputs))
            self.dX[:] = 0
        if self.dX.dtype != dtype:
            self.dX = self.dX.astype(dtype, copy=False)    
    
    def backward(self, W, dY, dW):
        (k,b,_) = dY.shape
        self.resize_bw(k, b, W.dtype)
        dW[:] = 0
        algo.convbw(W, self.X, dY, dW, self.dX)
        return self.dX
        
if __name__ == '__main__':
    import test
    ins = 2
    outs = 2
    window = 3
    layer = Convolution(ins,outs,3)
    test.layer(layer, float_t=np.float64)
    test.layer(layer, float_t=np.float32)
