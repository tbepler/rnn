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
    def __init__(self, n_in, n_topics, layers=[]):
        if len(layers) > 0:
            self.stack = LayeredLSTM(n_in, layers)
            n_in = layers[-1]
        else:
            self.stack = None
        self.top = LSTM(n_in, n_topics, gact=dampen, cact=dampen)

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

class Attention(object):
    def __init__(self, n):
        w = np.random.randn(n, n).astype(theano.config.floatX)
        orthogonal(w)
        self.W = theano.shared(w, borrow=True)

    @property
    def weights(self):
        return [self.W]

    def __getstate__(self):
        return {'W': self.W.get_values(borrow=True)}

    def __setstate__(self, state):
        self.W = theano.shared(state['W'], borrow=True)
        
    def __call__(self, L, R, mask=None):
        L = L / T.sqrt(T.sum(L**2, axis=-1, keepdims=True))
        R = R / T.sqrt(T.sum(R**2, axis=-1, keepdims=True))
        pivot = (L[:-2]+R[2:])
        pivot = T.concatenate([R[0:1], pivot, L[-2:-1]], axis=0)
        pivot = pivot / T.sqrt(T.sum(pivot**2, axis=-1, keepdims=True))
        W = T.dot(pivot, self.W)
        i = T.shape_padright(T.arange(L.shape[0]))
        I = T.shape_padright(T.shape_padright(i.T < i))
        J = T.shape_padright(T.shape_padright(i.T > i))
        A = L*I + R*J
        P = T.sum(A*W, axis=-1)
        i = T.arange(L.shape[0])
        P = T.set_subtensor(P[i,i], float('-inf'))
        if mask is not None:
            P += T.log(mask)
        P = softmax(P, axis=1)
        C = T.sum(A*T.shape_padright(P), axis=1)
        C = C / T.sqrt(T.sum(C**2, axis=-1, keepdims=True))
        return P, C
        
class CouplingLSTM(object):
    def __init__(self, n_in, n_components, layers=[], weights_r2=0.01, grad_clip=None):
        self.n_in = n_in
        self.n_components = n_components
        self.weights_r2 = weights_r2
        self.grad_clip = grad_clip
        self.forward = LayeredLSTM(n_in, layers+[n_components])
        self.backward = LayeredLSTM(n_in, layers+[n_components])
        self._attention = Attention(n_components)
        self.logit_decoder = Linear(n_components, n_in)

    def __getstate__(self):
        state = {}
        state['n_in'] = self.n_in
        state['n_components'] = self.n_components
        state['weights_r2'] = self.weights_r2
        state['grad_clip'] = self.grad_clip
        state['forward'] = self.forward
        state['backward'] = self.backward
        state['attention'] = self._attention
        state['logit_decoder'] = self.logit_decoder
        return state

    def __setstate__(self, state):
        self.n_in = state['n_in']
        self.n_components = state['n_components']
        self.weights_r2 = state['weights_r2']
        self.grad_clip = state['grad_clip']
        self.forward = state['forward']
        self.backward = state['backward']
        self._attention = state['attention']
        self.logit_decider = state['logit_decoder']

    @property
    def weights(self):
        return self.forward.weights + self.backward.weights + self._attention.weights + self.logit_decoder.weights

    def class_logprob(self, V):
        return logsoftmax(self.logit_decoder(V), axis=-1)

    def transform(self, X, mask=None):
        L = self.forward.scanl(X, mask=mask, clip=self.grad_clip)
        R = self.backward.scanr(X, mask=mask, clip=self.grad_clip)
        _, C = self._attention(L, R, mask=mask)
        return C

    def attention(self, X, mask=None):
        L = self.forward.scanl(X, mask=mask, clip=self.grad_clip)
        R = self.backward.scanr(X, mask=mask, clip=self.grad_clip)
        A, _ = self._attention(L, R, mask=mask)
        return A

    def prior(self, Z):
        i,j = T.mgrid[0:Z.shape[0],0:Z.shape[1]]
        I = Z.argmax(axis=-1)
        return T.set_subtensor(Z[i,j,I], 0).sum(axis=-1)

    def regularizer(self):
        for w in self.weights:
            loss += self.weights_r2*T.sum(w*w)
        return loss
    
    def _loss(self, X, mask=None, flank=0, regularize=True):
        n = X.shape[0]
        V = self.transform(X, mask=mask)
        X = X[flank:n-flank]
        if mask is not None:
            mask = mask[flank:n-flank]
        V = V[flank:n-flank]
        Yh = self.class_logprob(V)
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

        





