import ctypes
import numpy as np
import math

import rnn.kernel as algo
import rnn.initializers as init

class Linear(object):

    def __init__(self, input_size, output_size, initializer=init.xavier):
        self.inputs = input_size
        self.outputs = output_size
        self.initializer = initializer
        self.Y = None
        self.dX = None

    def weights(self, dtype=np.float64):
        W = np.zeros((self.inputs+1, self.outputs), dtype=dtype)
        self.initializer(W[1:])
        return W
        
    @property
    def shape(self):
        return (self.inputs+1, self.outputs)

    @property
    def size(self):
        return (self.inputs+1)*self.outputs

    def __getstate__(self):
        return self.inputs, self.outputs, self.initializer

    def __setstate__(self, x):
        self.inputs, self.outputs, self.initializer = x
        self.Y = None
        self.dX = None

    def reset(self):
        pass

    def advance(self):
        pass
            
    def resize_fw(self, k, b, dtype):
        if self.Y is None:
            self.Y = np.zeros((k,b,self.outputs), dtype=dtype)
        else:
            self.Y.resize((k,b,self.outputs), refcheck=False)
        if self.Y.dtype != dtype:
            self.Y = self.Y.astype(dtype, copy=False)    
            
    def forward(self, W, X, train=None):
        (k,b,_) = X.shape
        self.resize_fw(k, b, W.dtype)
        algo.linearfw(W, X, self.Y)
        self.X = X
        return self.Y

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
        algo.linearbw(W, self.X, dY, dW, self.dX)
        return self.dX
        
if __name__ == '__main__':
    import test
    ins = 2
    outs = 2
    layer = Linear(ins,outs)
    test.layer(layer, float_t=np.float64)
    test.layer(layer, float_t=np.float32)
