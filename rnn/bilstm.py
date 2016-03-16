import ctypes
import numpy as np
import math

import algorithms as algo
import initializers as init

class BiLSTM(object):

    def __init__(self, input_size, output_size, forget_bias=3, tau=float('inf')
                 , initializer = init.xavier):
        self.inputs = input_size
        self._outputs = output_size
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
        self._Y = None
        self.S = None
        self.dX = None
        self.dS = None

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
    def Y(self):
        return self._Y[1:-1]

    @property
    def outputs(self): return 2*self._outputs
        
    def advance(self):
        #bidirectional LSTM is not really foldable...
        if self._Y is not None:
            self._Y[0,:,:] = self._Y[-2,:,:]
            self._Y[-1] = 0
        if self.S is not None:
            self.S[0,:,:] = self.S[-2,:,:]
            self.S[-1] = 0

    def reset(self):
        if self._Y is not None:
            self._Y[:] = 0
        if self.S is not None:
            self.S[:] = 0    
            
    def resize_fwd(self, k, b, dtype):
        if self._Y is None:
            self._Y = np.zeros((k+2,b,2*self._outputs), dtype=dtype)
        else:
            self._Y.resize((k+2,b,2*self._outputs),refcheck=False)
        if self._Y.dtype != dtype:
            self._Y = self._Y.astype(dtype, copy=False)    
        if self.S is None:
            self.S = np.zeros((k+2,b,12*self._outputs), dtype=dtype)
        else:
            self.S.resize((k+2,b,12*self._outputs))
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
            self.dS = np.zeros((k+2,b,12*self._outputs), dtype=dtype)
        else:
            self.dS.resize((k+2,b,12*self._outputs))
            self.dS[:] = 0
        if self.dS.dtype != dtype:
            self.dS = self.dS.astype(dtype, copy=False)    
            
    def forward(self, W, X, train=None):
        (k,b,_) = X.shape
        #self.advance()
        self.resize_fwd(k, b, W.dtype)
        #algo.bilstmfw(W, X, self.S, self._Y)
        algo.lstmfw(W[0:self._outputs+self.inputs+1], X, self.S[0:-1,:,0:6*self._outputs]
                    , self._Y[0:-1,:,0:self._outputs])
        algo.lstmrfw(W[self._outputs+self.inputs+1:], X, self.S[1:,:,6*self._outputs:]
                     , self._Y[1:,:,self._outputs:])
        self.X = X
        return self._Y[1:-1]

    def backward(self, W, dY, dW):
        (k,b,_) = dY.shape
        self.resize_bwd(k, b, W.dtype)
        #dW[:] = 0
        #algo.bilstmbw(self.tau, W, self.X, self.S, self._Y, dY, dW, self.dX, self.dS)
        algo.lstmbw(self.tau, W[0:self._outputs+self.inputs+1], self.X
                    , self.S[0:-1,:,0:6*self._outputs], self._Y[0:-1,:,0:self._outputs]
                    , dY[:,:,0:self._outputs], dW[0:self._outputs+self.inputs+1]
                    , self.dX, self.dS[0:-1,:,0:6*self._outputs])
        algo.lstmrbw(self.tau, W[self._outputs+self.inputs+1:], self.X
                     , self.S[1:,:,6*self._outputs:], self._Y[1:,:,self._outputs:]
                     , dY[:,:,self._outputs:], dW[self._outputs+self.inputs+1:]
                     , self.dX, self.dS[1:,:,6*self._outputs:])
        return self.dX

if __name__ == '__main__':
    import test
    ins = 2
    outs = 2
    layer = BiLSTM(ins,outs)
    test.layer(layer, float_t=np.float64)
    test.layer(layer, float_t=np.float32)
