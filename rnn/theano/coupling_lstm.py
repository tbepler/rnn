import theano
import theano.tensor as T
import numpy as np

from lstm import LSTM, LayeredLSTM, LayeredBLSTM
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

def sigmoid(x):
    return T.nnet.sigmoid(x)
    #return T.tanh(x)/2 + 0.5

class LSTMStack(object):
    def __init__(self, n_in, n_components, layers=[], **kwargs):
        if len(layers) > 0:
            self.stack = LayeredLSTM(n_in, layers
                    , iact=sigmoid
                    , fact=sigmoid
                    , oact=sigmoid
                    , gact=T.tanh
                    , cact=T.tanh)
            n_in = layers[-1]
        else:
            self.stack = None
        self.top = LSTM(n_in, n_components
                , iact=sigmoid
                , fact=sigmoid
                , oact=sigmoid
                , gact=T.tanh
                , cact=T.tanh
                , **kwargs)

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

def normalize(X, axis=-1):
    Z = T.sqrt(T.sum(X**2, axis=axis, keepdims=True))
    Z = T.maximum(Z, 1e-38)
    return X / Z
    #R = T.log(abs(X))
    #R -= T.max(R, axis=axis, keepdims=True)
    #X = T.sgn(X)*T.exp(R)
    #Z = T.sqrt(T.sum(X**2, axis=axis, keepdims=True))
    #return X / T.maximum(Z, 1e-6)

class Attention(object):
    def __init__(self, n):
        w = np.random.randn(n, n).astype(theano.config.floatX)
        orthogonal(w)
        self.W = theano.shared(w, borrow=True)

    @property
    def weights(self):
        return [self.W]

    def __getstate__(self):
        return {'W': self.W.get_value(borrow=True)}

    def __setstate__(self, state):
        self.W = theano.shared(state['W'], borrow=True)
        
    def __call__(self, L, R, pivot=None, mask=None):
        if pivot is None:
            L = normalize(L)
            R = normalize(R)
            pivot = normalize(L[:-2]+R[2:])
            pivot = T.concatenate([R[1:2], pivot, L[-2:-1]], axis=0)
        W = T.dot(pivot, self.W)
        i = T.shape_padright(T.arange(L.shape[0]))
        I = T.shape_padright(T.shape_padright(i.T < i))
        J = T.shape_padright(T.shape_padright(i.T > i))
        A = L*I + R*J
        P = T.sum(A*W.dimshuffle(0,'x',1,2), axis=-1)
        #P = T.log1p(T.nnet.relu(P))
        i = T.arange(L.shape[0])
        P = T.set_subtensor(P[i,i], float('-inf'))
        if mask is not None:
            P += T.log(mask)
        #    P *= mask
        P = softmax(P, axis=1)
        C = T.sum(A*T.shape_padright(P), axis=1)
        #return P/T.maximum(P.sum(axis=1, keepdims=True), 1e-38), C
        return P, normalize(C)
        #return T.zeros(A.shape), pivot
        
class RecurrentAttention(object):
    def __init__(self, n_in, n_out):
        w = np.random.randn(n_out, n_out).astype(theano.config.floatX)
        orthogonal(w)
        self.W = theano.shared(w, borrow=True)
        u = np.random.randn(n_out, n_out).astype(theano.config.floatX)
        orthogonal(u)
        self.U = theano.shared(u, borrow=True)
        v = np.random.randn(n_out).astype(theano.config.floatX)
        v /= np.sqrt(np.sum(v**2))
        self.v = theano.shared(v, borrow=True)
        self.left = LSTMStack(n_in, n_out, []) 
        self.right = LSTMStack(n_in, n_out, [])

    @property
    def weights(self):
        return [self.W, self.U, self.v] + self.left.weights + self.right.weights

    def __getstate__(self):
        state = {}
        state['W'] = self.W.get_value(borrow=True)
        state['U'] = self.U.get_value(borrow=True)
        state['v'] = self.v.get_value(borrow=True)
        state['left'] = self.left
        state['right'] = self.right
        return state

    def __setstate__(self, state):
        self.W = theano.shared(state['W'], borrow=True)
        self.U = theano.shared(state['U'], borrow=True)
        self.v = theano.shared(state['v'], borrow=True)
        self.left = state['left']
        self.right = state['right']
        
    def decoder_state(self, X, mask=None):
        L, _ = self.left.scanl(X, mask=mask)
        R, _ = self.right.scanr(X, mask=mask)
        S = (L[:-2]+R[2:])/2
        return T.concatenate([R[1:2], S, L[-2:-1]], axis=0)

    def __call__(self, X, L, R, mask=None):
        i = T.shape_padright(T.arange(L.shape[0]))
        I = T.shape_padright(T.shape_padright(i.T < i))
        J = T.shape_padright(T.shape_padright(i.T > i))
        H = L*I + R*J
        S = self.decoder_state(X, mask=mask) 
        A = T.tanh(H.dot(self.W) + S.dot(self.U).dimshuffle(0, 'x', 1, 2)).dot(self.v)
        i = T.arange(L.shape[0])
        A = T.set_subtensor(A[i,i], float('-inf'))
        if mask is not None:
            A += T.log(mask)
        A = softmax(A, axis=1)
        C = T.sum(H*T.shape_padright(A), axis=1)
        return A, C

class CouplingLSTM(object):
    def __init__(self, n_in, n_components, layers=[], atten_layers=[], weights_r2=0.01, grad_clip=None):
        self.n_in = n_in
        self.n_components = n_components
        self.weights_r2 = weights_r2
        self.grad_clip = grad_clip
        self.forward = LSTMStack(n_in, n_components, layers)
        self.backward = LSTMStack(n_in, n_components, layers)
        #self.fw_pivot = LayeredLSTM(n_in, layers+[n_components])
        #self.bw_pivot = LayeredLSTM(n_in, layers+[n_components])
        #self._attention = RecurrentAttention(n_components, atten_layers)
        self._attention = RecurrentAttention(n_in, n_components)
        self.logit_decoder = Linear(n_components, n_in)

    def __getstate__(self):
        state = {}
        state['n_in'] = self.n_in
        state['n_components'] = self.n_components
        state['weights_r2'] = self.weights_r2
        state['grad_clip'] = self.grad_clip
        state['forward'] = self.forward
        state['backward'] = self.backward
        #state['fw_pivot'] = self.fw_pivot
        #state['bw_pivot'] = self.bw_pivot
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
        #self.fw_pivot = state['fw_pivot']
        #self.bw_pivot = state['bw_pivot']
        self._attention = state['attention']
        self.logit_decider = state['logit_decoder']

    @property
    def weights(self):
        #return self.forward.weights + self.backward.weights + self.fw_pivot.weights + self.bw_pivot.weights + self._attention.weights + self.logit_decoder.weights
        return self.forward.weights + self.backward.weights + self._attention.weights + self.logit_decoder.weights

    def class_logprob(self, V):
        return logsoftmax(self.logit_decoder(V), axis=-1)

    def _pivot(self, X, mask=None):
        L, _, _, _ = self.fw_pivot.scanl(X, mask=mask, clip=self.grad_clip)
        R, _, _, _ = self.bw_pivot.scanr(X, mask=mask, clip=self.grad_clip)
        pivot = (L[:-2]+R[2:])
        pivot = T.concatenate([R[1:2], pivot, L[-2:-1]], axis=0)
        return normalize(pivot)

    def transform(self, X, mask=None):
        #L,_ = self.forward.scanl(X, mask=mask, clip=self.grad_clip, activation=normalize)
        #R,_ = self.backward.scanr(X, mask=mask, clip=self.grad_clip, activation=normalize)
        L,_ = self.forward.scanl(X, mask=mask, clip=self.grad_clip)
        R,_ = self.backward.scanr(X, mask=mask, clip=self.grad_clip)
        #pivot = self._pivot(X, mask=mask)
        #_, C = self._attention(L, R, pivot=pivot, mask=mask)
        _, C = self._attention(X, L, R, mask=mask)
        return C

    def attention(self, X, mask=None):
        #L,_ = self.forward.scanl(X, mask=mask, clip=self.grad_clip, activation=normalize)
        #R,_ = self.backward.scanr(X, mask=mask, clip=self.grad_clip, activation=normalize)
        L,_ = self.forward.scanl(X, mask=mask, clip=self.grad_clip)
        R,_ = self.backward.scanr(X, mask=mask, clip=self.grad_clip)
        #pivot = self._pivot(X, mask=mask)
        #A, _ = self._attention(L, R, pivot=pivot, mask=mask)
        A, _ = self._attention(X, L, R, mask=mask)
        return A

    def prior(self, Z):
        i,j = T.mgrid[0:Z.shape[0],0:Z.shape[1]]
        I = Z.argmax(axis=-1)
        return T.set_subtensor(Z[i,j,I], 0).sum(axis=-1)

    def regularizer(self):
        loss = 0
        for w in self.weights:
            loss += self.weights_r2*T.sum(w**2)
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
        loss = T.sum(L)/mask.sum()
        if regularize:
            loss += self.regularizer()
        return loss, L, C

    def loss(self, X, mask=None, flank=0):
        _, L, C = self._loss(X, mask=mask, flank=flank)
        return L, C

    def gradient(self, X, mask=None, flank=0):
        loss, L, C = self._loss(X, mask=mask, flank=flank)
        gW = theano.grad(loss, self.weights, disconnected_inputs='warn')
        #gW = [theano.printing.Print('Gradient {}: '.format(i))(gW[i]) for i in xrange(len(gW))]
        return gW, [L.sum(axis=[0,1]),C.sum(axis=[0,1])]

        





