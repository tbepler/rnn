import numpy as np
import math
import ctypes

import rnn.kernel as algo

class SoftmaxCrossEntropy(object):

    def __init__(self):
        self.Y = None
        self.dX = None

    def __getstate__(self):
        return True

    def __setstate__(self, x):
        self.Y = None
        self.dX = None

    def weights(self, dtype=np.float64):
        return np.zeros(0, dtype=dtype)

    @property
    def shape(self): return 0

    @property
    def size(self): return 0

    @property
    def dW(self): return None
        
    def reset(self):
        pass

    def advance(self):
        pass
    
    def resize_fw(self, X):
        if self.Y is None:
            self.Y = np.zeros(X.shape, dtype=X.dtype)
        else:
            if X.dtype != self.Y.dtype:
                self.Y = self.Y.astype(X.dtype, copy=False)
            if X.shape != self.Y.shape:
                self.Y.resize(X.shape, refcheck=False)
    
    def forward(self, W, X, train=None):
        self.resize_fw(X)
        algo.softmaxfw(X, self.Y)
        return self.Y

    def resize_bw(self):
        if self.dX is None:
            self.dX = np.zeros(self.Y.shape, dtype=self.Y.dtype)
        else:
            if self.dX.dtype != self.Y.dtype:
                self.dX = self.dX.astype(self.Y.dtype, copy=False)
            if self.dX.shape != self.Y.shape:
                self.dX.resize(self.Y.shape)
    
    def backward(self, W, Y, dW):
        self.resize_bw()
        algo.entmaxbw(self.Y, Y, self.dX)
        return self.dX


def cross_entropy( yh, y ):
    return algo.centfw(yh, y)

class CrossEntropyError:
    def __init__(self, dtype):
        self.dtype = dtype

    def init(self, shape, dtype=None):
        (k,b,n) = shape
        self.y = np.random.randint(n, size=(k,b))
        return self

    def __call__(self, yh):
        return cross_entropy(yh, self.y)

    def grad(self, yh):
        return self.y

def test():
    import test
    size = 4
    layer = SoftmaxCrossEntropy()
    
    error_f = CrossEntropyError(np.int32)
    test.layer(layer, ins=size, outs=size, err_f=error_f, float_t = np.float64)
    test.layer(layer, ins=size, outs=size, err_f=error_f, float_t = np.float32)

    error_f = CrossEntropyError(np.int64)
    test.layer(layer, ins=size, outs=size, err_f=error_f, float_t=np.float64)
    test.layer(layer, ins=size, outs=size, err_f=error_f, float_t=np.float32)
    
if __name__ == '__main__':
    test()

