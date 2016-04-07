import ctypes
import numpy as np
import math

import rnn.kernel as algo

class Embedding(object):

    def __init__(self, input_size, output_size, dtype=np.float64):
        self.inputs = input_size
        self.outputs = output_size
        self._dtype = dtype
        self.W = np.random.randn(self.inputs, self.outputs)/math.sqrt(self.inputs)
        self.W = self.W.astype(dtype, copy=False)
        self.Y = None
        self.dW = np.zeros_like(self.W)

    @property
    def weights(self):
        return self.W

    @weights.setter
    def weights(self, w):
        assert w.shape == self.shape
        self.W = w

    @property
    def grad(self):
        return self.dW

    @grad.setter
    def grad(self, dw):
        assert dw.shape == self.shape
        self.dW = dw
        
    def __getstate__(self):
        return self.inputs, self.outputs, self._dtype, self.W

    def __setstate__(self, x):
        self.inputs, self.outputs, self._dtype, self.W = x
        self.Y = None
        self.dW = np.zeros_like(self.W)

    @property
    def dtype(self): return self._dtype

    @dtype.setter
    def dtype(self, t):
        self._dtype = t
        self.W = self.W.astype(t, copy=False)
        if not self.Y is None:
            self.Y = self.Y.astype(t, copy=False)
        if not self.dW is None:
            self.dW = self.dW.astype(t, copy=False)
        if not self.solver is None:
            self.solver.dtype = t

    def reset(self):
        pass

    def advance(self):
        pass

    def resize_fw(self, k, b):
        if self.Y is None:
            self.Y = np.zeros((k, b, self.outputs), dtype=self.dtype)
        else:
            self.Y.resize((k,b,self.outputs), refcheck=False)
        
    def forward(self, X, train=None):
        (k,b) = X.shape
        self.resize_fw(k,b)
        algo.embedfw(self.W, X, self.Y)
        self.X = X
        return self.Y
    
    def backward(self, dY):
        (k,b,_) = dY.shape
        self.dW[:] = 0
        algo.embedbw(self.X, dY, self.dW)
        return None
        

def test():
    import test
    ins = 4
    outs = 4
    layer = Embedding(ins,outs)
    init_x = lambda k,b,n: np.random.randint(n, size=(k,b))
    
    test.layer(layer, type_x=np.int32, init_x=init_x, float_t=np.float64)
    test.layer(layer, type_x=np.int64, init_x=init_x, float_t=np.float64)

    test.layer(layer, type_x=np.int32, init_x=init_x, float_t=np.float32)
    test.layer(layer, type_x=np.int64, init_x=init_x, float_t=np.float32)
    
if __name__ == '__main__':
    test()
