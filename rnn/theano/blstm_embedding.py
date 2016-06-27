import theano
import theano.tensor as T
import numpy as np

from lstm import LSTM, LayeredLSTM
from linear import Linear
from softmax import logsoftmax, softmax, logsumexp
from loss import cross_entropy, confusion
from rnn.initializers import orthogonal

def ident(x):
    return x

def dampen(x):
    #transform x as sign(x)*ln(|x|+1)
    # preserves full range, but dampens the gradient
    return T.sgn(x)*T.log(abs(x)+1)

class LSTMStack(object):
    def __init__(self, n_in, n_components, layers=[], **kwargs):
        if len(layers) > 0:
            self.stack = LayeredLSTM(n_in, layers)
            n_in = layers[-1]
        else:
            self.stack = None
        self.top = LSTM(n_in, n_components, **kwargs)

    @property
    def weights(self):
        ws = []
        if self.stack is not None:
            ws.extend(self.stack.weights)
        ws.extend(self.top.weights)
        return ws

    def scanl(self, X, **kwargs):
        if self.stack is not None:
            X, _, _, _ = self.stack.scanl(X, **kwargs)
        return self.top.scanl(X, **kwargs)

    def scanr(self, X, **kwargs):
        if self.stack is not None:
            X, _, _, _ = self.stack.scanr(X, **kwargs)
        return self.top.scanr(X, **kwargs)

def normalize(x, axis=-1):
    Z = T.sqrt(T.sum(x**2, axis=axis, keepdims=True))
    Z = T.maximum(Z, 1e-39) #about the closest to zero float32 can go
    return x/Z

class BlstmEmbed(object):
    def __init__(self, n_in, n_components, hidden_units=[], l2_reg=0.01, type='real', grad_clip=None):
        self.n_in = n_in
        self.n_components = n_components
        self.l2_reg = l2_reg
        self.grad_clip = grad_clip
        if type == 'real':
            self.forward = LSTMStack(n_in, n_components, layers=hidden_units)
            self.backward = LSTMStack(n_in, n_components, layers=hidden_units)
        elif type == 'non-neg':
            from activation import fast_sigmoid
            self.forward = LSTMStack(n_in, n_components, layers=hidden_units
                    , cact=fast_sigmoid
                    , gact=fast_sigmoid)
            self.backward = LSTMStack(n_in, n_components, layers=hidden_units
                    , cact=fast_sigmoid
                    , gact=fast_sigmoid)
        else:
            raise "Type: '{}' not supported".format(type)
        self.type = type
        self.logit_decoder = Linear(n_components, n_in)

    def __getstate__(self):
        state = {}
        state['n_in'] = self.n_in
        state['n_components'] = self.n_components
        state['l2_reg'] = self.l2_reg
        state['grad_clip'] = self.grad_clip
        state['forward'] = self.forward
        state['backward'] = self.backward
        state['type'] = self.type
        state['logit_decoder'] = self.logit_decoder
        return state

    def __setstate__(self, state):
        self.n_in = state['n_in']
        self.n_components = state['n_components']
        self.l2_reg = state['l2_reg']
        self.grad_clip = state['grad_clip']
        self.forward = state['forward']
        self.backward = state['backward']
        self.type = state['type']
        self.logit_decoder = state['logit_decoder']

    @property
    def weights(self):
        return self.forward.weights + self.backward.weights + self.logit_decoder.weights

    def combine(self, L, R):
        Z = normalize(L[:-2] + R[2:])
        return T.concatenate([L[1:2], Z, R[-2:-1]], axis=0)
        
    def class_logprob(self, V):
        return logsoftmax(self.logit_decoder(V), axis=-1)

    def transform(self, X, mask=None):
        L, _ = self.forward.scanl(X, mask=mask, clip=self.grad_clip, activation=normalize)
        R, _ = self.backward.scanr(X, mask=mask, clip=self.grad_clip, activation=normalize)
        return self.combine(L, R)

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
        Yh = self.class_logprob(Z)
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