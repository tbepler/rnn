import theano
import theano.tensor as T
import numpy as np

from lstm import LSTM, LayeredLSTM
from linear import Linear
from softmax import logsoftmax, softmax, logsumexp
from loss import cross_entropy, confusion
from rnn.initializers import orthogonal
from activation import fast_sigmoid, fast_tanh

def ident(x):
    return x

def dampen(x):
    #transform x as sign(x)*ln(|x|+1)
    # preserves full range, but dampens the gradient
    return T.sgn(x)*T.log(abs(x)+1)

def sigmoid(x):
    return (T.tanh(x)+1)/2

class LSTMStack(object):
    def __init__(self, n_in, n_components, layers=[], sigmoid=fast_sigmoid, tanh=fast_tanh, **kwargs):
        if len(layers) > 0:
            self.stack = LayeredLSTM(n_in, layers
                    , iact=sigmoid
                    , fact=sigmoid
                    , oact=sigmoid
                    , gact=tanh
                    , cact=tanh)
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
            X, _, _, _ = self.stack.scanl(X)
        return self.top.scanl(X, **kwargs)

    def scanr(self, X, **kwargs):
        if self.stack is not None:
            X, _, _, _ = self.stack.scanr(X)
        return self.top.scanr(X, **kwargs)

class BlstmEmbed(object):
    def __init__(self, n_in, n_components, hidden_units=[], l2_reg=0.01, type='real', grad_clip=None
            , sigmoid=fast_sigmoid, tanh=fast_tanh, scale=ident):
        self.n_in = n_in
        self.n_components = n_components
        self.l2_reg = l2_reg
        self.grad_clip = grad_clip
        self.scale = scale
        if type == 'real':
            self.forward = LSTMStack(n_in, n_components, layers=hidden_units
                    , sigmoid=sigmoid
                    , tanh=tanh
                    , iact=sigmoid
                    , fact=sigmoid
                    , oact=sigmoid
                    , gact=tanh
                    , cact=tanh)
            self.backward = LSTMStack(n_in, n_components, layers=hidden_units
                    , sigmoid=sigmoid
                    , tanh=tanh
                    , iact=sigmoid
                    , fact=sigmoid
                    , oact=sigmoid
                    , gact=tanh
                    , cact=tanh)
        elif type == 'non-neg':
            self.forward = LSTMStack(n_in, n_components, layers=hidden_units
                    , sigmoid=sigmoid
                    , tanh=tanh
                    , iact=sigmoid
                    , fact=sigmoid
                    , oact=sigmoid
                    , cact=sigmoid
                    , gact=sigmoid)
            self.backward = LSTMStack(n_in, n_components, layers=hidden_units
                    , sigmoid=sigmoid
                    , tanh=tanh
                    , iact=sigmoid
                    , fact=sigmoid
                    , oact=sigmoid
                    , cact=sigmoid
                    , gact=sigmoid)
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
        state['scale'] = self.scale
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
        self.scale = state.get('scale', ident)

    @property
    def weights(self):
        return self.forward.weights + self.backward.weights + self.logit_decoder.weights

    def combine(self, L, R):
        #Z = normalize(L[:-2] + R[2:])
        Z = self.scale((L[:-2] + R[2:])/2)
        return T.concatenate([R[1:2], Z, L[-2:-1]], axis=0)
        
    def class_logprob(self, V):
        return logsoftmax(self.logit_decoder(V), axis=-1)

    def transform(self, X, mask=None, units='output'):
        if units == 'output':
            L, _ = self.forward.scanl(X, mask=mask, clip=self.grad_clip, activation=self.scale)
            R, _ = self.backward.scanr(X, mask=mask, clip=self.grad_clip, activation=self.scale)
        elif units == 'state':
            _, L = self.forward.scanl(X, mask=mask, clip=self.grad_clip, activation=self.scale)
            L = self.scale(self.forward.top.cact(L))
            _, R = self.backward.scanr(X, mask=mask, clip=self.grad_clip, activation=self.scale)
            R = self.scale(self.backward.top.cact(R))
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
