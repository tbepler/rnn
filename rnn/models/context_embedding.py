import theano
import theano.tensor as T
import numpy as np

import rnn.theano.solvers as solvers

def null_func(*args, **kwargs):
    pass

class ContextEmbedding(object):
    def __init__(self, encoder, optimizer=solvers.RMSprop(0.01), batch_size=100, length=-1, flank=0):
        self.encoder = encoder
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.length = length
        self.flank = flank
        self.theano_setup()

    def __getstate__(self):
        state = {}
        state['encoder'] = self.encoder
        state['optimizer'] = self.optimizer
        state['batch_size'] = self.batch_size
        state['length'] = self.length
        state['flank'] = self.flank
        return state

    def __setstate__(self, state):
        self.__init__(state['encoder'], optimizer=state['optimizer']
                , batch_size=state['batch_size']
                , length=state['length']
                , flank=state['flank'])

    def theano_setup(self):
        self.theano = object()
        self.theano.X = T.matrix()
        self.theano.mask = T.matrix(dtype='int8')

    def split(self, data, length=None, flank=None, keep_index=True):
        if length is None:
            length = self.length
        if flank is None:
            flank = self.flank
        index = 0
        for x in data:
            mask = None
            if flank > 0:
                n = len(x)
                mask = np.zeros(n+2*flank, dtype=np.int8)
                mask[flank:-flank] = 1
                x_flanked = np.zeros(n+2*flank, dtype=x.dtype)
                x_flanked[flank:-flank] = x
                x = x_flanked
            if length > 0:
                m = length + 2*flank
                for i in xrange(0, n, length):
                    start = i
                    len = min(n-i, length)
                    idx = (index, start, len)
                    if mask is not None and keep_index:
                        yield idx, x[i:i+m], mask[i:i+m]
                    elif mask is not None and not keep_index:
                        yield x[i:i+m], mask[i:i+m]
                    elif keep_index:
                        yield idx, x[i:i+m]
                    else:
                        yield x[i:i+m]
            else:
                idx = (index, 0, len(x))
                if mask is not None and keep_index:
                    yield idx, x, mask
                elif mask is not None and not keep_index:
                    yield x, mask
                elif keep_index:
                    yield idx, x
                else:
                    yield x
            index += 1

    def make_minibatch(self, xs):
        b = len(xs)
        cols = zip(*xs)
        if len(cols) > 2:
            index, xs, mask = cols
        else:
            index, xs = cols
            mask = None
        n = max(len(x) for x in xs)
        X = np.zeros((n,b), dtype=xs[0].dtype)
        M = np.zeros((n,b), dtype=np.int8)
        for i in xrange(b):
            k = len(xs[i])
            X[:k,i] = xs[i]
            if mask is not None: 
                M[:k,i] = mask[i]
            else:
                M[:k,i] = 1
        return index, X, M

    def minibatched(self, data):
        xs = []
        for x in data:
            if len(xs) >= self.batch_size:
                yield self.make_minibatch(xs)
                del xs[:]
            xs.append(x)
        if len(xs) > 0:
            yield self.make_minibatch(xs)

    def _compile_loss_minibatch(self):
        if not hasattr(self, 'loss_minibatch'):
            flank = T.iscalar()
            X = self.theano.X
            mask = self.theano.mask
            n = X.shape[0]
            res = self.encoder.loss(X, mask)
            if flank > 0:
                res = [r[flank:n-flank] for r in res]
            res = [T.sum(r, axis=0) for r in res]
            self.loss_minibatch = theano.function([X, mask, flank], res)  

    def loss_iter(self, data, callback=null_func):
        total = 0
        if hasattr(data, '__len__'):
            total = sum(len(x) for x in data)
            callback(0, 'loss')
        n = 0
        i = 0
        L = None
        for index, X, M in self.minibatched(self.split(data)):
            res = self.loss_minibatch(X, M, self.flank)
            if L is None:
                L = [0 for _ in res]
            for j in xrange(len(index)):
                n += index[j][2]
                if index[j][0] > i:
                    yield L
                    L = [0 for _ in res]
                    i = index[j][0]
                L = [l+r[j] for l,r in zip(L,res)]
            if total > 0:
                callback(float(n)/total, 'loss')
        yield L

    def loss_accum(self, data, callback=null_func):
        total = 0
        if hasattr(data, '__len__'):
            total = sum(len(x) for x in data)
            callback(0, 'loss')
        n = 0
        i = 0
        L = None
        for index, X, M in self.minibatched(self.split(data)):
            res = self.loss_minibatch(X, M, self.flank)
            if L is None:
                L = [0 for _ in res]
            for i in xrange(len(res)):
                L[i] += res[i].sum(axis=0)
            n += sum(idx[2] for idx in index)
            if total > 0:
                callback(float(n)/total, 'loss')
        return L

    def loss(self, data, callback=null_func, axis=None):
        if axis == 0:
            return self.loss_iter(data, callback=callback)
        else:
            return self.loss_accum(data, callback=callback)

    def fit_steps(self, minibatches, max_iters=100):
        gws, res = self.encoder.gradient(self.theano.X, self.theano.mask, flank=self.flank)
        weights = self.encoder.weights
        return self.solver(minibatches, [self.theano.X, self.theano.mask], res, weights, grads=gws, max_iters=max_iters)        

    def fit(self, train, validate=None, max_iters=100, callback=null_func):
        from rnn.minibatcher import BatchIter
        train_data = list(self.split(train, keep_index=False))
        train_minibatches = BatchIter(train_data, self.batch_size)
        steps = self.fit_steps(train_minibatches, max_iters=max_iters)
        callback(0, 'fit')
        train_res = None
        for it, res in steps:
            if train_res is None:
                train_res = [0 for _ in res]
            for i in xrange(len(res)):
                train_res[i] += res[i]
            if it % 1 == 0:
                if validate is not None:
                    val_res = self.loss(validate, callback=callback)
                    yield val_res + train_res
                else:
                    yield train_res
                for i in xrange(len(train_res)):
                    train_res[i] = 0
            callback(it%1, 'fit')

    def _compile_transform_minibatch(self):
        if not hasattr(self, 'transform_minibatch'):
            flank = T.iscalar()
            X = self.theano.X
            mask = self.theano.mask
            n = X.shape[0]
            Z = self.encoder.transform(X, mask)
            if flank > 0:
                Z = Z[flank:n-flank]
            self.transform_minibatch = theano.function([X, mask, flank], Z)  

    def transform(self, data, callback=null_func):
        self._compile_transform_minibatch()
        total = 0
        if hasattr(data, '__len__'):
            total = sum(len(x) for x in data)
            callback(0, 'transform')
        n = 0
        for index,X,M in self.minibatched(self.split(data)):
            Z = self.transform_minibatch(X, M, self.flank)
            n += sum(idx[2] for idx in index)
            yield index, Z
            if total > 0:
                callback(float(n)/total, 'transform')
    
