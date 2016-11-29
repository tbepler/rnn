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
        X = X.dimshuffle(1,0,2) ## reshape from (batch,length,features)
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
        return F.dimshuffle(1,0,2) ## reshape back to (batch,length,features)
    
    def backward(self, X):
        X = X.dimshuffle(1,0,2) ## reshape from (batch,length,features)
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
        return B.dimshuffle(1,0,2) ## reshape back to (batch,length,features)

    def posterior(self, X):
        F = self.forward(X)
        B = self.backward(X)
        return logsoftmax(F+B, axis=-1)


    def viterbi(self, X, mask=None, **kwargs):
        X = X.dimshuffle(1,0,2) ## reshape from (batch,length,features)
        if mask is not None:
            mask = mask.dimshuffle(1,0)

        ## compute the transition probabilities given the input
        if self.use_bias:
            inits = logsoftmax(T.dot(X[0], self.w_init[1:]) + self.w_init[0], axis=-1)
            trans = logsoftmax(T.dot(X[1:], self.w_trans[:,1:]) + self.w_trans[:,0], axis=-1)
        else:
            inits = logsoftmax(T.dot(X[0], self.w_init), axis=-1)
            trans = logsoftmax(T.dot(X[1:], self.w_trans), axis=-1)

        ## viterbi step
        def step(A, p0):
            p0 = T.shape_padright(p0)
            P = A + p0
            tt = P.argmax(axis=-2)
            pt = P.max(axis=-2)
            return pt, tt
        [P, Tb], _ = theano.scan(step, trans, [inits, None], *kwargs)
        P = T.concatenate([T.shape_padleft(inits), P], axis=0)
        Tb = T.concatenate([Tb, T.zeros_like(Tb[0:1])], axis=0)

        J = T.arange(Tb.shape[1])
        ## traceback the ML path
        def step(pt, tt, mt, i0):
            # if position is masked
            masked_it = T.zeros_like(i0) - 1
            masked_p = T.zeros_like(pt[:,0])
            # if i0 is undefined (-1)
            start_it = pt.argmax(axis=-1)
            start_p = pt.max(axis=-1)
            # if continuing a path
            cont_it = tt[J,i0]
            cont_p = pt[J,cont_it]
            # switch on the conditions
            _it = T.switch(T.eq(i0, -1), start_it, cont_it)
            _p = T.switch(T.eq(i0, -1), start_p, cont_p)
            it = T.switch(T.eq(mt, 0), masked_it, _it)
            p = T.switch(T.eq(mt, 0), masked_p, _p)
            return it, p
        start = T.zeros_like(Tb[0,:,0]) - 1
        [path, probs], _ = theano.scan(step, [P[::-1], Tb[::-1], mask[::-1]], [start, None])
        path = path[::-1].dimshuffle(1, 0) # reshape back
        probs = probs[::-1].dimshuffle(1, 0)

        return path, probs
        
        
    def logprob(self, X, Y, mask=None, **kwargs):
        X = X.dimshuffle(1,0,2) ## reshape from (batch,length,features)
        Y = Y.dimshuffle(1,0)
        if mask is not None:
            mask = mask.dimshuffle(1,0)

        if self.use_bias:
            inits = logsoftmax(T.dot(X[0], self.w_init[1:]) + self.w_init[0], axis=-1)
            trans = logsoftmax(T.dot(X[1:], self.w_trans[:,1:]) + self.w_trans[:,0], axis=-1)
        else:
            inits = logsoftmax(T.dot(X[0], self.w_init), axis=-1)
            trans = logsoftmax(T.dot(X[1:], self.w_trans), axis=-1)
        k,b = Y.shape
        mesh = T.mgrid[0:k-1,0:b]
        i,j = mesh[0], mesh[1]
        if mask is not None: #integrate over the masked positions
            def step(A, yt, mt, p0):
                p0 = T.shape_padright(p0)
                B = A + p0
                observed = B[T.arange(yt.shape[0]),yt]
                unobserved = logsumexp(B, axis=-2)
                mt = T.shape_padright(mt)
                pt = T.switch(T.eq(mt, 0), unobserved, observed)
                return pt
            P, _ = theano.scan(step, [trans, Y[:-1], mask[:-1]], inits, **kwargs)
        else:
            P = trans[i,j,Y[:-1]]
        P = T.concatenate([T.shape_padleft(inits), P], axis=0)
        return logsoftmax(P, axis=-1).dimshuffle(1,0,2) ##reshape back


        
            

