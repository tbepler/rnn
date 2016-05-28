import theano
import theano.tensor as T

from lstm import LSTM
from linear import Linear
from softmax import logsoftmax
from loss import cross_entropy, confusion
from rnn.initializers import orthogonal
from autoencoder import NullNoise

class DirichletTransition(object):
    def __init__(self, n, init=orthogonal, dtype=theano.config.floatX):
        w_trans = np.random.randn(n, n).astype(dtype)
        init(w_trans)
        w_init = np.zeros(n, dtype=dtype)
        self.w_trans = theano.shared(w_trans, borrow=True)
        self.w_init = theano.shared(w_init, borrow=True)

    def __getstate__(self):
        state = {}
        state['w_trans'] = self.w_trans.get_value(borrow=True)
        state['w_init'] = self.w_init.get_value(borrow=True)

    def __setstate__(self, state):
        self.w_trans = theano.shared(state['w_trans'], borrow=True)
        self.w_init = theano.shared(state['w_init'], borrow=True)

    @property
    def weights(self):
        return [self.w_trans, self.w_init]

    def __call__(self, X):
        A = T.dot(X[:-1], self.w_trans)
        A = T.exp(T.concatenate([w_init, A], axis=0))
        B = T.sum(T.gammaln(A), axis=-1) - T.gammaln(T.sum(A, axis=-1))
        L = T.dot(A-1, X.dimshuffle(0, 2, 1)) - B

class Emmission(object):
    def __init__(self, m, n, init=orthogonal, dtype=theano.config.floatX):
        w = np.random.randn(m, n).astype(dtype)
        init(w)
        self.w = theano.shared(w, borrow=True)

    def __getstate__(self):
        return {'w': self.w.get_value(borrow=True)}

    def __setstate__(self, state):
        self.w = theano.shared(state['w'], borrow=True)

    def __call__(self, X):
        P = T.dot(T.exp(X), softmax(self.w))
        P = T.log(P/P.sum(axis=-1,keepdims=True))
        return P

class TopicLSTM(object):
    def __init__(self, n_in, units, n_topics, sparsity=0, noise=NullNoise()):
        self.forward = LSTM(n_in, units)
        self.backward = LSTM(units, n_topics)
        self.trans = DirichletTransition(n_topics)
        self.emit = Emmission(n_topics, n_in)
        self.sparsity = sparsity
        self.noise = noise

    @property
    def weights(self):
        return self.forward.weights + self.backward.weights + self.trans.weights + self.emit.weights

    def transform(self, X, mask=None):
        Z_f = self.forward.scanl(X, mask=mask)
        Z = self.backward.scanr(Z_f, mask=None, activation=logsoftmax)
        return Z

    def loss(self, X, mask=None, flank=0, Z=None):
        if Z is None:
            Z = self.transform(self.noise(X), mask=mask)
        Tr = self.trans(Z)
        E = self.emit(Z)
        L = cross_entropy(T.shape_padright(Tr) + E, X)
        C = confusion(T.argmax(E,axis=-1), X, E.shape[-1])
        if mask is not None:
            L *= T.shape_padright(mask)
            C *= T.shape_padright(T.shape_padright(mask))
        n = X.shape[0]
        return L[flank:n-flank], C[flank:n-flank]

    def gradient(self, X, mask=None, flank=0):
        Z = self.transform(self.noise(X), mask=mask)
        L, C = self.loss(X, mask=mask, flank=flank, Z=Z)
        loss = T.sum(L)
        n = Z.shape[0]
        if self.sparcity > 0:
            R = self.sparcity*Z
            if mask is not None:
                R *= T.shape_padright(R)
            loss += T.sum(R[flank:n-flank])
        gW = theano.grad(loss, self.weights)
        return gW, [L,C]

        





