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
        pivot = (L[:-2]+R[2:])/2
        pivot = T.concatenate([R[0:1], pivot, L[-2:-1]], axis=0)
        W = T.dot(pivot, self.W)
        i = T.shape_padright(T.arange(L.shape[0]))
        I = T.shape_padright(T.shape_padright(i.T < i))
        J = T.shape_padright(T.shape_padright(i.T > i))
        A = L*I + R*J
        A = T.sum(A*W, axis=-1)
        

        

class CouplingLSTM(object):
    def __init__(self, n_in, n_components, layers=[], weights_r2=0.01, grad_clip=None):
        self.n_in = n_in
        self.n_components = n_components
        self.weights_r2 = weights_r2
        self.grad_clip = grad_clip
        self.forward = LayeredLSTM(n_in, layers+[n_components])
        self.backward = LayeredLSTM(n_in, layers+[n_components])
        w = np.random.randn(n_topics, n_components).astype(theano.config.floatX)
        orthogonal(w)
        self.topic_matrix = theano.shared(w, borrow=True)
        self.logit_decoder = Linear(n_components, n_in)

    def __getstate__(self):
        state = {}
        state['n_in'] = self.n_in
        state['n_topics'] = self.n_topics
        state['n_components'] = self.n_components
        state['sparsity'] = self.sparsity
        state['unscaled_topic_r2'] = self.unscaled_topic_r2
        state['weights_r2'] = self.weights_r2
        state['topic_orth_r2'] = self.topic_orth_r2
        state['grad_clip'] = self.grad_clip
        state['forward'] = self.forward
        state['backward'] = self.backward
        state['topic_matrix'] = self.topic_matrix.get_value(borrow=True)
        state['logit_decoder'] = self.logit_decoder
        return state

    def __setstate__(self, state):
        self.n_in = state['n_in']
        self.n_topics = state['n_topics']
        self.n_components = state['n_components']
        self.sparsity = state['sparsity']
        self.unscaled_topic_r2 = state['unscaled_topic_r2']
        self.weights_r2 = state['weights_r2']
        self.topic_orth_r2 = state.get('topic_orth_r2', 0)
        self.grad_clip = state['grad_clip']
        self.forward = state['forward']
        self.backward = state['backward']
        self.topic_matrix = theano.shared(state['topic_matrix'], borrow=True)
        self.logit_decider = state['logit_decoder']

    @property
    def weights(self):
        return self.forward.weights + self.backward.weights + [self.topic_matrix] + self.logit_decoder.weights

    def fuse(self, x, y):
        return x+y

    def unscaled_topic_mixture(self, X, mask=None, clip=None):
        #Y_f, C_f, _, _ = self.forward.scanl(X, mask=mask, clip=clip)
        #Y_b, C_b, _, _ = self.backward.scanr(X, mask=mask, clip=clip)
        Y_f, C_f = self.forward.scanl(X, mask=mask, clip=clip)
        Y_b, C_b = self.backward.scanr(X, mask=mask, clip=clip)
        Z = self.fuse(Y_f[:-2], Y_b[2:])
        Z = T.concatenate([Y_b[0:1], Z, Y_f[-2:-1]], axis=0)
        return Z

    def position_vector(self, P):
        W = self.topic_matrix/T.sqrt(T.sum(self.topic_matrix**2, axis=-1, keepdims=True))
        X = T.dot(P, W)
        X = X/T.sqrt(T.sum(X**2, axis=-1, keepdims=True))
        return X
        
    def class_logprob(self, V):
        return logsoftmax(self.logit_decoder(V), axis=-1)

    def transform(self, X, mask=None, topics=True, vectors=False):
        P = softmax(self.unscaled_topic_mixture(X, mask=mask))
        V = self.position_vector(P)
        if topics and not vectors:
            return P
        elif not topics and vectors:
            return V
        else:
            return [P, V]

    def prior(self, Z):
        i,j = T.mgrid[0:Z.shape[0],0:Z.shape[1]]
        I = Z.argmax(axis=-1)
        return T.set_subtensor(Z[i,j,I], 0).sum(axis=-1)

    def regularizer(self, Z, m, mask=None):
        if mask is not None:
            loss = self.sparsity*(self.prior(Z)*mask).sum() # / m
            loss += self.unscaled_topic_r2*((Z**2).sum(axis=-1)*mask).sum() # / m
        else:
            loss = self.sparsity*self.prior(Z).sum() # / m
            loss += self.unscaled_topic_r2*(Z**2).sum() # / m
        for w in self.weights:
            loss += self.weights_r2*T.sum(w*w)
        if self.topic_orth_r2 > 0:
            U = T.dot(self.topic_matrix, self.topic_matrix.T)
            loss += T.sum((U - T.eye(U.shape[0]))**2)
        return loss

    
    def _loss(self, X, mask=None, flank=0, clip=None, regularize=True):
        n = X.shape[0]
        Z = self.unscaled_topic_mixture(X, mask=mask, clip=clip)[flank:n-flank]
        X = X[flank:n-flank]
        mask = mask[flank:n-flank]
        P = softmax(Z)
        V = self.position_vector(P)
        Yh = self.class_logprob(V)
        L = cross_entropy(Yh, X)
        C = confusion(T.argmax(Yh,axis=-1), X, Yh.shape[-1])
        if mask is not None:
            L *= T.shape_padright(mask)
            C *= T.shape_padright(T.shape_padright(mask))
        m = mask.sum() if mask is not None else L.size
        loss = T.sum(L) # / m
        if regularize:
            loss += self.regularizer(Z, m, mask=mask)
        return loss, L, C

    def loss(self, X, mask=None, flank=0):
        _, L, C = self._loss(X, mask=mask, flank=flank)
        return L, C

    def gradient(self, X, mask=None, flank=0):
        loss, L, C = self._loss(X, mask=mask, flank=flank, clip=self.grad_clip)
        gW = theano.grad(loss, self.weights, disconnected_inputs='warn')
        return gW, [L.sum(axis=[0,1]),C.sum(axis=[0,1])]

        





