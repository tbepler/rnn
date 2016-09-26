import theano
import theano.tensor as T
import numpy as np

from rnn.theano.softmax import logsoftmax, logsumexp
from rnn.theano.loss import cross_entropy, confusion, accuracy
from rnn.initializers import orthogonal

class Loss(object):
    def __call__(self, crf, X, Y, mask=None, flank=0):
        Yh = self.decode(crf, X, Y)
        L = self.loss(Yh, Y)
        C = confusion(T.argmax(Yh,axis=-1), Y, Yh.shape[-1])
        if mask is not None:
            L *= T.shape_padright(mask)
            C *= T.shape_padright(T.shape_padright(mask))
        n = Yh.shape[0]
        return L[flank:n-flank], C[flank:n-flank]

class PosteriorCrossEntropy(Loss):
    def decode(self, crf, X, Y):
        return crf.posterior(X)

    def loss(self, Yh, Y):
        return cross_entropy(Yh, Y)

class LikelihoodCrossEntropy(Loss):
    def decode(self, crf, X, Y):
        return crf.logprob(X, Y)

    def loss(self, Yh, Y):
        return cross_entropy(Yh, Y)

class PosteriorAccuracy(Loss):
    def __init__(self, step=1):
        self.step = step

    def decode(self, crf, X, Y):
        return crf.posterior(X)

    def loss(self, Yh, Y):
        return accuracy(Yh, Y, step=self.step)

class LikelihoodAccuracy(Loss):
    def __init__(self, step=1):
        self.step = step

    def decode(self, crf, X, Y):
        return crf.logprob(X, Y)

    def loss(self, Yh, Y):
        return accuracy(Yh, Y, step=self.step)

class CRF(object):
    def __init__(self, ins, labels, init=orthogonal, name=None, dtype=theano.config.floatX
                , loss=LikelihoodCrossEntropy(), random=np.random, use_bias=True):
        self.use_bias = use_bias
        if use_bias:
            w_trans = random.randn(labels, 1+ins, labels).astype(dtype)
            w_trans[:,0] = 0
            for i in range(labels):
                init(w_trans[i, 1:])
            w_init = random.randn(1+ins, labels).astype(dtype)
            w_init[0] = 0
            init(w_init[1:])
        else:
            w_trans = random.randn(labels, ins, labels).astype(dtype)
            for i in range(labels):
                init(w_trans[i])
            w_init = random.randn(ins, labels).astype(dtype)
            init(w_init)
        self.w_trans = theano.shared(w_trans, borrow=True)
        self.w_init = theano.shared(w_init, borrow=True)
        self._loss = loss

    def __getstate__(self):
        state = {}
        state['use_bias'] = self.use_bias
        state['w_trans'] = self.w_trans.get_value(borrow=True)
        state['w_init'] = self.w_init.get_value(borrow=True)
        state['loss'] = self._loss
        return state

    def __setstate__(self, state):
        self.use_bias = state.get('use_bias', True)
        self.w_trans = theano.shared(state['w_trans'], borrow=True)
        self.w_init = theano.shared(state['w_init'], borrow=True)
        self._loss = state['loss']

    @property
    def shared(self):
        return [self.w_trans, self.w_init]

    @property
    def weights(self):
        if self.use_bias:
            return [self.w_trans[:,1:], self.w_init[1:]]
        return [self.w_trans, self.w_init]

    @property
    def bias(self):
        if self.use_bias:
            return [self.w_trans[:,0], self.w_init[0]]
        return []

    @shared.setter
    def shared(self, ws):
        self.w_trans.set_value(ws[0])
        self.w_init.set_value(ws[1])

    def loss(self, X, Y, **kwargs):
        return self._loss(self, X, Y, **kwargs)

    def forward(self, X):
        if self.use_bias:
            inits = logsoftmax(T.dot(X[0], self.w_init[1:]) + self.w_init[0], axis=-1)
            trans = logsoftmax(T.dot(X[1:], self.w_trans[:,1:]) + self.w_trans[:,0], axis=-1)
        else:
            inits = logsoftmax(T.dot(X[0], self.w_init), axis=-1)
            trans = logsoftmax(T.dot(X[1:], self.w_trans), axis=-1)
        def step(A, x0):
            x0 = T.shape_padright(x0)
            xt = logsumexp(A+x0, axis=-2) 
            return xt
        F, _ = theano.scan(step, trans, inits)
        F = T.concatenate([T.shape_padleft(inits), F], axis=0)
        return F
    
    def backward(self, X):
        if self.use_bias:
            trans = logsoftmax(T.dot(X[1:], self.w_trans[:,1:]) + self.w_trans[:,0], axis=-1)
        else:
            trans = logsoftmax(T.dot(X[1:], self.w_trans), axis=-1)
        def step(A, xt):
            xt = xt.dimshuffle(0, 'x', 1)
            x0 = logsumexp(A+xt, axis=-1)
            return x0
        b_end = T.zeros(trans.shape[1:-1], dtype=X.dtype)
        B, _ = theano.scan(step, trans[::-1], b_end)
        B = T.concatenate([B[::-1], T.shape_padleft(b_end)], axis=0)
        return B

    def posterior(self, X):
        F = self.forward(X)
        B = self.backward(X)
        return logsoftmax(F+B, axis=-1)
        
    def logprob(self, X, Y):
        if self.use_bias:
            inits = T.dot(X[0], self.w_init[1:]) + self.w_init[0]
            trans = T.dot(X[1:], self.w_trans[:,1:]) + self.w_trans[:,0]
        else:
            inits = T.dot(X[0], self.w_init)
            trans = T.dot(X[1:], self.w_trans)
        k,b = Y.shape
        mesh = T.mgrid[0:k-1,0:b]
        i,j = mesh[0], mesh[1]
        num = T.concatenate([T.shape_padleft(inits), trans[i,j,Y[:-1]]], axis=0)
        return logsoftmax(num, axis=-1)


        
            

