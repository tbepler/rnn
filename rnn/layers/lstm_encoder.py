import ctypes
import numpy as np
import math

import rnn.kernel as algo
import rnn.initializers as init

class LSTMEncoder(object):

    def __init__(self, input_size, output_size, forget_bias=3, tau=float('inf')
                 , initializer = init.xavier):
        self.inputs = input_size
        self.outputs = output_size
        self.forget_bias = forget_bias
        self.tau = tau
        self.initializer = initializer
        self._Y = None
        self.S = None
        self.dX = None
        self.dS = None

    def __getstate__(self):
        x = {}
        x['inputs'] = self.inputs
        x['outputs'] = self.outputs
        x['forget_bias'] = self.forget_bias
        x['tau'] = self.tau
        x['init'] = self.initializer
        return x

    def __setstate__(self, x):
        if type(x) is tuple: #for backwards compatability
            if len(x) == 6:
                (self.inputs, self.outputs, self.forget_bias, self.tau
                 , self.initializer, _) = x
            elif len(x) == 5:
                (self.inputs, self.outputs, self.forget_bias, self.tau
                 , self.initializer) = x
            elif len(x) == 4:
                (self.inputs, self.outputs, self.forget_bias, self.tau) = x
                self.initializer = init.xavier
        else:
             self.inputs = x.get('inputs', 0)
             self.outputs = x.get('outputs', 0)
             self.forget_bias = x.get('forget_bias', 3)
             self.tau = x.get('tau', float('inf'))
             self.initializer = x.get('init', init.xavier) 
        self._Y = None
        self.S = None
        self.dX = None
        self.dS = None

    def weights(self, dtype=np.float64):
        W = np.zeros((self.inputs+self.outputs+1, 4*self.outputs), dtype=dtype)
        self.initializer(W[1:])
        W[0,self.outputs:2*self.outputs] = self.forget_bias
        return W
        
    @property
    def shape(self): return (self.inputs+self.outputs+1, 4*self.outputs)

    @property
    def size(self): return (self.inputs+self.outputs+1)*4*self.outputs

    @property
    def Y(self):
        return self._Y[-1:]
        
    def advance(self):
        if self._Y is not None:
            self._Y[0,:,:] = self._Y[-1,:,:]
        if self.S is not None:
            self.S[0,:,:] = self.S[-1,:,:]    

    def reset(self):
        if self._Y is not None:
            self._Y[:] = 0
        if self.S is not None:
            self.S[:] = 0    
            
    def resize_fwd(self, k, b, dtype):
        if self._Y is None:
            self._Y = np.zeros((k+1,b,self.outputs), dtype=dtype)
        else:
            self._Y.resize((k+1,b,self.outputs),refcheck=False)
        if self._Y.dtype != dtype:
            self._Y = self._Y.astype(dtype, copy=False)    
        if self.S is None:
            self.S = np.zeros((k+1,b,6*self.outputs), dtype=dtype)
        else:
            self.S.resize((k+1,b,6*self.outputs))
        if self.S.dtype != dtype:
            self.S = self.S.astype(dtype, copy=False)    

    def resize_bwd(self, k, b, dtype): 
        if self.dX is None:
            self.dX = np.zeros((k,b,self.inputs), dtype=dtype)
        else:
            self.dX.resize((k,b,self.inputs))
            self.dX[:] = 0
        if self.dX.dtype != dtype:
            self.dX = self.dX.astype(dtype, copy=False)
        if self.dS is None:
            self.dS = np.zeros((k+1,b,6*self.outputs), dtype=dtype)
        else:
            self.dS.resize((k+1,b,6*self.outputs))
            self.dS[:] = 0
        if self.dS.dtype != dtype:
            self.dS = self.dS.astype(dtype, copy=False)    
            
    def forward(self, W, X, train=None):
        (k,b,_) = X.shape
        #self.advance()
        self.resize_fwd(k, b, W.dtype)
        algo.lstmfw(W, X, self.S, self._Y)
        self.X = X
        return self._Y[-1:]

    def backward(self, W, dY, dW):
        (_,b,_) = dY.shape
        k = self._Y.shape[0]-1
        self.resize_bwd(k, b, W.dtype)
        #dW[:] = 0
        algo.lstmencbw(self.tau, W, self.X, self.S, self._Y, dY, dW, self.dX, self.dS)
        return self.dX

if __name__ == '__main__':
    import test
    ins = 2
    outs = 2
    layer = LSTMEncoder(ins,outs)
    test.layer(layer, float_t=np.float64)
    test.layer(layer, float_t=np.float32)
