import theano
import theano.tensor as T
import numpy as np

from lstm import LSTM
from linear import Linear
from softmax import logsoftmax, softmax, logsumexp
from loss import cross_entropy, confusion
from rnn.theano.activation import normalize
from rnn.initializers import orthogonal

class LSTMEncoder(object):
    def __init__(self, n_in, units, iact=T.nnet.sigmoid, fact=T.nnet.sigmoid, oact=T.nnet.sigmoid, gact=T.tanh
            , cact=T.tanh, scaling=normalize, grad_clip=None):
        units = units // 2
        self.left = LSTM(n_in, units, iact=iact, fact=fact, oact=oact, gact=gact, cact=cact)
        self.right = LSTM(n_in, units, iact=iact, fact=fact, oact=oact, gact=gact, cact=cact)
        self.scaling = scaling
        self.grad_clip = grad_clip

    @property
    def weights(self):
        return self.left.weights + self.right.weights

    def combine(self, L, R):
        return self.scaling(T.concatenate([L, R], axis=-1))

    def fix_dims(self, x, Y):
        shape = Y.shape[1:]
        z = T.zeros(shape, dtype=x.dtype)
        return T.shape_padleft(x+z)

    def transform(self, X, mask=None, units='output'):
        if units == 'output':
            L, _ = self.left.scanl(X, mask=mask, clip=self.grad_clip)
            L = T.concatenate([self.fix_dims(self.left.y0, L), L], axis=0)[:-1] 
            R, _ = self.right.scanr(X, mask=mask, clip=self.grad_clip)
            R = T.concatenate([R, self.fix_dims(self.right.y0, R)], axis=0)[1:]
        elif units == 'state':
            _, L = self.left.scanl(X, mask=mask, clip=self.grad_clip)
            L = T.concatenate([self.fix_dims(self.left.c0, L), L], axis=0)[:-1] 
            L = self.left.cact(L)
            _, R = self.right.scanr(X, mask=mask, clip=self.grad_clip)
            R = T.concatenate([R, self.fix_dims(self.right.c0, R)], axis=0)[1:]
            R = self.right.cact(R)
        return self.combine(L, R)

class DeepDecoder(object):
    def __init__(self, n_in, units, hidden=[], hidden_activation=T.tanh):
        self.hidden_layers = []
        for n in hidden:
            self.hidden_layers.append(Linear(n_in, n))
            n_in = n
        self.linear = Linear(n_in, units)
        self.hidden_activation = hidden_activation

    @property
    def weights(self):
        ws = []
        for layer in self.hidden_layers:
            ws.extend(layer.weights)
        ws.extend(self.linear.weights)
        return ws

    def decode(self, X):
        for layer in self.hidden_layers:
            X = self.hidden_activation(layer(X))
        return logsoftmax(self.linear(X))

class RecurrentEmbed(object):
    def __init__(self, encoder, decoder, l2_reg=0.01):
        self.encoder = encoder
        self.decoder = decoder
        self.l2_reg = l2_reg

    def __getstate__(self):
        state = {}
        state['encoder'] = self.encoder
        state['decoder'] = self.decoder
        state['l2_reg'] = self.l2_reg
        return state

    def __setstate__(self, state):
        self.encoder = state['encoder']
        self.decoder = state['decoder']
        self.l2_reg = state['l2_reg']

    @property
    def weights(self):
        return self.encoder.weights + self.decoder.weights

    def transform(self, X, mask=None, **kwargs):
        return self.encoder.transform(X, mask=mask, **kwargs)

    def prior(self, Z):
        i,j = T.mgrid[0:Z.shape[0],0:Z.shape[1]]
        I = Z.argmax(axis=-1)
        return T.set_subtensor(Z[i,j,I], 0).sum(axis=-1)

    def regularizer(self):
        loss = 0
        for w in self.weights:
            loss += self.l2_reg*T.sum(w*w)
        return loss
    
    def _loss(self, X, mask=None, flank=0, regularize=True):
        n = X.shape[0]
        Z = self.transform(X, mask=mask)[flank:n-flank]
        X = X[flank:n-flank]
        mask = mask[flank:n-flank]
        Yh = self.decoder.decode(Z)
        L = cross_entropy(Yh, X)
        C = confusion(T.argmax(Yh,axis=-1), X, Yh.shape[-1])
        if mask is not None:
            L *= T.shape_padright(mask)
            C *= T.shape_padright(T.shape_padright(mask))
        loss = T.sum(L)
        if regularize:
            loss += self.regularizer()
        return loss, L, C

    def loss(self, X, mask=None, flank=0):
        _, L, C = self._loss(X, mask=mask, flank=flank)
        return L, C

    def gradient(self, X, mask=None, flank=0):
        loss, L, C = self._loss(X, mask=mask, flank=flank)
        gW = theano.grad(loss, self.weights, disconnected_inputs='warn')
        return gW, [L.sum(axis=[0,1]),C.sum(axis=[0,1])]
