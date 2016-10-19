import theano
import theano.tensor as T
import numpy as np

from softmax import logsoftmax, logsumexp
from loss import cross_entropy, confusion, accuracy, multiconfusion
from initializers import orthogonal

class Loss(object):
    def __call__(self, crf, X, Y, mask=None, flank=0):
        Yh = self.decode(crf, X, Y)
        L = self.loss(Yh, Y)
        C = confusion(T.argmax(Yh,axis=-1), Y, Yh.shape[-1])
        if mask is not None:
            L *= T.shape_padright(mask)
            C *= T.shape_padright(T.shape_padright(mask))
        n = Yh.shape[0]
        return L[flank:n-flank], C[flank:n-flank], Yh, Y

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
            , loss=LikelihoodCrossEntropy(), label_sects=None):
        w_trans = np.random.randn(labels, 1+ins, labels).astype(dtype)
        w_trans[:,0] = 0
        for i in xrange(labels):
            init(w_trans[i, 1:])
        w_init = np.random.randn(1+ins, labels).astype(dtype)
        w_init[0] = 0
        init(w_init[1:])
        self.w_trans = theano.shared(w_trans, borrow=True)
        self.w_init = theano.shared(w_init, borrow=True)
        self._loss = loss
        self.labels = labels
        self.label_sects = label_sects

    def __getstate__(self):
        state = {}
        state['w_trans'] = self.w_trans.get_value(borrow=True)
        state['w_init'] = self.w_init.get_value(borrow=True)
        state['loss'] = self._loss
        return state

    def __setstate__(self, state):
        self.w_trans = theano.shared(state['w_trans'], borrow=True)
        self.w_init = theano.shared(state['w_init'], borrow=True)
        self._loss = state['loss']

    @property
    def weights(self):
        return [self.w_trans, self.w_init]

    @weights.setter
    def weights(self, ws):
        self.w_trans.set_value(ws[0])
        self.w_init.set_value(ws[1])

    def update(self, change, history=None):
        adds, dels = change
        print "adds = %s" % adds
        print "dels = %s" % dels
        if adds != []:
        # Adding unit
            trans = self.w_trans.get_value()
            init = self.w_init.get_value()
            #print trans
            #print init
            for add in adds:
                print trans.shape
                print init.shape
                print add[0]
                init = np.insert(init, add[0] + 1, np.random.rand(add[1], self.labels), axis = 0)
                trans = np.insert(trans, [add[0] + 1], np.random.rand(self.labels, add[1], self.labels), axis = 1)
                #print trans.shape
                #print init.shape
            self.w_trans.set_value(trans)
            self.w_init.set_value(init)

            if history is not None:
                for hist in history:
                    hist_trans = hist[0].get_value()
                    hist_init = hist[1].get_value()
                    for add in adds:
                        hist_trans = np.insert(hist_trans, [add[0] + 1], np.random.rand(self.labels, add[1], self.labels), axis = 1)
                        hist_init = np.insert(hist_init, add[0] + 1, np.random.rand(add[1], self.labels), axis = 0)
                    hist[0].set_value(hist_trans)
                    hist[1].set_value(hist_init)
        if dels != []:
        # Removing unit
            trans = self.w_trans.get_value()
            init = self.w_init.get_value()
            for remove in dels:
                trans = np.delete(trans, remove + 1, 1)
                init = np.delete(init, remove + 1, 0)
            self.w_trans.set_value(trans)
            self.w_init.set_value(init)
            if history is not None:
                for hist in history:
                    hist_trans = hist[0].get_value()
                    hist_init = hist[1].get_value()
                    for remove in dels:
                        print remove
                        print hist_trans.shape
                        hist_trans = np.delete(hist_trans, remove + 1, 1)
                        hist_init = np.delete(hist_init, remove + 1, 0)
                    hist[0].set_value(hist_trans)
                    hist[1].set_value(hist_init)

    def loss(self, X, Y, **kwargs):
        return self._loss(self, X, Y, **kwargs)

    def forward(self, X):
        inits = logsoftmax(T.dot(X[0], self.w_init[1:]) + self.w_init[0], axis=-1)
        trans = logsoftmax(T.dot(X[1:], self.w_trans[:,1:]) + self.w_trans[:,0], axis=-1)
        def step(A, x0):
            x0 = T.shape_padright(x0)
            xt = logsumexp(A+x0, axis=-2) 
            return xt
        F = theano.scan(step, trans, inits)[0]
        F = T.concatenate([T.shape_padleft(inits), F], axis=0)
        return F
    
    def backward(self, X):
        trans = logsoftmax(T.dot(X[1:], self.w_trans[:,1:]) + self.w_trans[:,0], axis=-1)
        def step(A, xt):
            xt = xt.dimshuffle(0, 'x', 1)
            x0 = logsumexp(A+xt, axis=-1)
            return x0
        b_end = T.zeros(trans.shape[1:-1])
        B = theano.scan(step, trans[::-1], b_end)[0]
        B = T.concatenate([B[::-1], T.shape_padleft(b_end)], axis=0)
        return B

    def posterior(self, X):
        F = self.forward(X)
        B = self.backward(X)
        return logsoftmax(F+B, axis=-1)
        
    def logprob(self, X, Y):
        inits = T.dot(X[0], self.w_init[1:]) + self.w_init[0]
        trans = T.dot(X[1:], self.w_trans[:,1:]) + self.w_trans[:,0]
        k,b = Y.shape
        mesh = T.mgrid[0:k-1,0:b]
        i,j = mesh[0], mesh[1]
        num = T.concatenate([T.shape_padleft(inits), trans[i,j,Y[:-1]]], axis=0)
        return logsoftmax(num, axis=-1)

class MultiCRF(CRF):
    def __init__(self, ins, labels, label_sects, init=orthogonal, name=None, dtype=theano.config.floatX
                , loss=LikelihoodCrossEntropy()):
        CRF.__init__(self, ins, labels, init, name, dtype, loss)
        self.crf = []
        self.label_sects = label_sects
        for i in range(label_sects):
            new_crf = CRF(ins, labels, init=init, name=name, dtype=dtype, loss=loss)
            self.crf += [new_crf]

    def loss(self, X, Y, **kwargs):
        axis = 1
        Ysect = [Y[:,i,:] for i in range(self.label_sects)]
        L = []
        C = []
        Yh = []
        for i in range(self.label_sects):
            l,c,yh,y = self.crf[i].loss(X, Ysect[i], **kwargs)
            L += [l]
            #C += [c]
            Yh += [yh]
        L = sum(L)
        #C = T.stack(C, axis=0)
        Yh = T.stack(Yh, axis=1)
        #C = multiconfusion(T.argmax(Yh,axis=-1), Y, Yh.shape[-1])
        #C = T.shape_padright(T.shape_padright(T.eq(Y,Yh)))
        C = T.eq(Y,T.argmax(Yh, axis=-1))
        return L, C, Yh, Y

    def update(self, change, history=None):
        for crf in self.crf:
            crf.update(change, history)

    @property
    def weights(self):
        weights = []
        for crf in self.crf:
            weights += crf.weights
        return weights
