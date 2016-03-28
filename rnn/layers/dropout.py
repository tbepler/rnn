import numpy as np
import random

import rnn.kernel as algo

class Dropout(object):

    def __init__(self, p):
        self.p = p
        self.mask = None
        self.Y = None

    def weights(self, dtype=np.float64):
        return np.zeros(0,dtype=dtype)

    @property
    def shape(self): return 0

    @property
    def size(self): return 0
        
    def __getstate__(self):
        return self.p

    def __setstate__(self, x):
        self.p = x
        self.mask = None
        self.Y = None
        
    def reset(self):
        pass

    def advance(self):
        pass

    def resize(self, shape, dtype):
        if self.mask is None:
            self.mask = np.zeros(shape, dtype=np.int32)
        else:
            self.mask.resize(shape)
        if self.Y is None:
            self.Y = np.zeros(shape, dtype=dtype)
        else:
            self.Y.resize(shape, refcheck=False)
        if self.Y.dtype != dtype:
            self.Y = self.Y.astype(dtype, copy=False)
    
    def forward(self, W, X, train=False):
        if train:
            self.resize(X.shape, X.dtype)
            np.copyto(self.Y, X)
            algo.dropout(self.p, self.Y, self.mask, 1) #1 for forward
            return self.Y
        return X

    def backward(self, W, dY, dW):
        algo.dropout(self.p, dY, self.mask, 0) #0 for backward
        return dY
        
