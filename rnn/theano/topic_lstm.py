import theano
import theano.tensor as T
import numpy as np

from lstm import LSTM, LayeredLSTM
from linear import Linear
from softmax import logsoftmax, softmax, logsumexp
from loss import cross_entropy, confusion
from rnn.initializers import orthogonal
from autoencoder import NullNoise

class DirichletTransition(object):
    def __init__(self, n, init=orthogonal, dtype=theano.config.floatX, eps=0.001, r1=0, r2=1):
        w_trans = np.random.randn(n, n).astype(dtype)
        init(w_trans)
        w_init = np.zeros(n, dtype=dtype)
        self.w_trans = theano.shared(w_trans, borrow=True)
        self.w_init = theano.shared(w_init, borrow=True)
        self.eps = eps
        self.r1 = r1
        self.r2 = r2

    def __getstate__(self):
        state = {}
        state['w_trans'] = self.w_trans.get_value(borrow=True)
        state['w_init'] = self.w_init.get_value(borrow=True)
        state['eps'] = self.eps
        state['r1'] = self.r1
        state['r2'] = self.r2

    def __setstate__(self, state):
        self.w_trans = theano.shared(state['w_trans'], borrow=True)
        self.w_init = theano.shared(state['w_init'], borrow=True)
        self.eps = state['eps']
        self.r1 = state['r1']
        self.r2 = state['r2']

    @property
    def weights(self):
        return [self.w_trans, self.w_init]

    def regularizer(self):
        r2 = ((self.w_trans**2).sum()+(self.w_init**2).sum())*self.r2
        r1 = (self.w_trans.norm(1)+self.w_init.norm(1))*self.r1
        return r2 + r1

    def __call__(self, X):
        #A = T.exp(T.dot(T.exp(X[:-1]), self.w_trans)) + self.eps
        #B = T.sum(T.gammaln(A), axis=-1) - T.gammaln(T.sum(A, axis=-1))
        #L = T.sum((A-1)*X[1:], axis=-1) - B
        A = softmax(T.dot(T.exp(X[:-1]), self.w_trans))
        L = T.sum(A*X[1:], axis=-1)
        #A_init = T.exp(self.w_init) + self.eps
        #B_init = T.sum(T.gammaln(A_init)) - T.gammaln(T.sum(A_init))
        #L_init = T.sum((A_init-1)*X[0], axis=-1) - B_init
        A_init = softmax(self.w_init)
        L_init = T.sum(A_init*X[0], axis=-1)
        return T.concatenate([T.shape_padleft(L_init), L], axis=0)

class Emmission(object):
    def __init__(self, m, n, init=orthogonal, dtype=theano.config.floatX):
        w = np.random.randn(m, n).astype(dtype)
        init(w)
        self.w = theano.shared(w, borrow=True)
        
    @property
    def weights(self):
        return [self.w]

    def __getstate__(self):
        return {'w': self.w.get_value(borrow=True)}

    def __setstate__(self, state):
        self.w = theano.shared(state['w'], borrow=True)

    def __call__(self, X):
        return logsoftmax(T.dot(X, self.w), axis=-1)
        #L = T.shape_padright(T.log(X)) + logsoftmax(self.w)
        #L = logsumexp(L, axis=-2)
        #return logsoftmax(L, axis=-1)
        #L = T.dot(X, softmax(self.w))
        #L = L/L.sum(axis=-1, keepdims=True)
        #return T.log(L)

def ident(x):
    return x

class TopicLSTM(object):
    def __init__(self, n_in, units, sparsity=0):
        self.forward = LayeredLSTM(n_in, units
                , cact=ident
                )
        self.backward = LayeredLSTM(n_in, units
                , cact=ident
                )
        self.emit = Emmission(units[-1], n_in)
        self.sparsity = sparsity
        self.n_in = n_in

    @property
    def weights(self):
        return self.forward.weights + self.backward.weights + self.emit.weights

    def fuse(self, x, y):
        return (x+y)/2.0

    def transform(self, X, mask=None):
        Y_f, C_f, _, _ = self.forward.scanl(X, mask=mask)
        Y_b, C_b, _, _ = self.backward.scanr(X, mask=mask)
        Z = self.fuse(Y_f[:-2], Y_b[2:])
        Z = T.concatenate([Y_b[0:1], Z, Y_f[-2:-1]], axis=0)
        return Z

    def loss(self, X, mask=None, flank=0, Z=None):
        if Z is None:
            Y_f, C_f, _, _ = self.forward.scanl(X, mask=mask)
            Y_b, C_b, _, _ = self.backward.scanr(X, mask=mask)
            Z = self.fuse(Y_f[:-2], Y_b[2:])
            Z = T.concatenate([Y_b[0:1], Z, Y_f[-2:-1]], axis=0)
        E = self.emit(Z)
        L = cross_entropy(E, X)
        C = confusion(T.argmax(E,axis=-1), X, E.shape[-1])
        if mask is not None:
            L *= T.shape_padright(mask)
            C *= T.shape_padright(T.shape_padright(mask))
        n = X.shape[0]
        return L[flank:n-flank], C[flank:n-flank]

    def gradient(self, X, mask=None, flank=0):
        Y_f, C_f, _, _ = self.forward.scanl(X, mask=mask, clip=1.0)
        Y_b, C_b, _, _ = self.backward.scanr(X, mask=mask, clip=1.0)
        Z = self.fuse(Y_f[:-2], Y_b[2:])
        Z = T.concatenate([Y_b[0:1], Z, Y_f[-2:-1]], axis=0)
        n = Z.shape[0]
        L, C = self.loss(X, mask=mask, flank=flank, Z=Z)
        loss = T.sum(L) #/ self.n_in
        if self.sparsity > 0:
            R = self.fuse(C_f[flank:n-flank], C_b[flank:n-flank])*T.shape_padright(mask[flank:n-flank])
            loss += self.sparsity*T.sum(abs(R))
        for w in self.weights:
            loss += 0.01*T.sum(w**2)
        gW = theano.grad(loss, self.weights, disconnected_inputs='warn')
        return gW, [L.sum(axis=[0,1]),C.sum(axis=[0,1])]

        





