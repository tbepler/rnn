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

    def __getstate__(self):
        state = {}
        state['encoder'] = self.encoder
        state['optimizer'] = self.optimizer
        state['batch_size'] = self.batch_size
        state['length'] = self.length
        state['flank'] = self.flank
        return state

    def __setstate__(self, state):
        self.encoder = state['encoder']
        self.optimizer = state['optimizer']
        self.batch_size = state['batch_size']
        self.length = state['length']
        self.flank = state['flank']

    def _theano_mask(self):
        return T.matrix(dtype='int8')

    def split(self, data, length=None, flank=None, keep_index=True):
        if length is None:
            length = self.length
        if flank is None:
            flank = self.flank
        index = 0
        for x in data:
            mask = None
            n = len(x)
            if flank > 0:
                mask = np.zeros(n+2*flank, dtype=np.int8)
                mask[flank:-flank] = 1
                x_flanked = np.zeros(n+2*flank, dtype=x.dtype)
                x_flanked[flank:-flank] = x
                x = x_flanked
            if length > 0:
                m = length + 2*flank
                for i in xrange(0, n, length):
                    start = i
                    l = min(n-i, length)
                    idx = (index, start, l)
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

    def _compile_loss_minibatch(self, dtype):
        if not hasattr(self, 'loss_minibatch'):
            flank = T.iscalar()
            X = T.matrix(dtype=dtype)
            mask = self._theano_mask()
            n = X.shape[0]
            res = self.encoder.loss(X, mask)
            res = [r[flank:n-flank] for r in res]
            res = [T.sum(r, axis=0) for r in res]
            return theano.function([X, mask, flank], res)  
        else:
            return self.loss_minibatch

    def loss_iter(self, data, callback=null_func):
        loss_minibatch = self._compile_loss_minibatch(dtype=data[0].dtype)
        total = 0
        if hasattr(data, '__len__'):
            total = sum(len(x) for x in data)
            callback(0, 'loss')
        n = 0
        i = 0
        L = None
        for index, X, M in self.minibatched(self.split(data)):
            res = loss_minibatch(X, M, self.flank)
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
        loss_minibatch = self._compile_loss_minibatch(dtype=data[0].dtype)
        total = 0
        if hasattr(data, '__len__'):
            total = sum(len(x) for x in data)
            callback(0, 'loss')
        n = 0
        i = 0
        L = None
        for index, X, M in self.minibatched(self.split(data)):
            res = loss_minibatch(X, M, self.flank)
            if L is None:
                L = [0 for _ in res]
            for i in xrange(len(res)):
                L[i] += res[i].sum(axis=0)
            n += sum(idx[2] for idx in index)
            if total > 0:
                callback(float(n)/total, 'loss')
        return L

    def loss(self, data, callback=null_func, axis=None, **kwargs):
        if axis == 0:
            return self.loss_iter(data, callback=callback)
        else:
            return self.loss_accum(data, callback=callback)

    def fit_steps(self, minibatches, max_iters=100):
        X = T.matrix(dtype=minibatches.dtype)
        mask = self._theano_mask()
        gws, res = self.encoder.gradient(X, mask, flank=self.flank)
        weights = self.encoder.weights
        return self.optimizer(minibatches, [X, mask], res, weights, grads=gws, max_iters=max_iters)        

    def fit(self, train, validate=None, max_iters=100, callback=null_func, yield_every=1.0, **kwargs):
        from rnn.minibatcher import BatchIter
        train_data = list(self.split(train, keep_index=False))
        train_minibatches = BatchIter(train_data, self.batch_size)
        steps = self.fit_steps(train_minibatches, max_iters=max_iters)
        callback(0, 'fit')
        train_res = None
        last_yield = 0.0
        for it, res in steps:
            if train_res is None:
                train_res = [0 for _ in res]
            for i in xrange(len(res)):
                train_res[i] += res[i]
            if it - last_yield >= yield_every:
                last_yield = it
                if validate is not None:
                    val_res = self.loss(validate, callback=callback)
                    yield it, val_res + train_res
                else:
                    yield it, train_res
                for i in xrange(len(train_res)):
                    train_res[i] = 0
            callback(it%1, 'fit')

    def _compile_transform_minibatch(self, dtype, **kwargs):
        if not hasattr(self, 'transform_minibatch'):
            flank = T.iscalar()
            X = T.matrix(dtype=dtype)
            mask = self._theano_mask()
            n = X.shape[0]
            Z = self.encoder.transform(X, mask, **kwargs)
            Z = Z[flank:n-flank]
            return theano.function([X, mask, flank], Z)  
        else:
            return self.transform_minibatch

    def transform(self, data, callback=null_func, **kwargs):
        transform_minibatch = self._compile_transform_minibatch(data[0].dtype, **kwargs)
        total = 0
        if hasattr(data, '__len__'):
            total = sum(len(x) for x in data)
            callback(0, 'transform')
        n = 0
        for index,X,M in self.minibatched(self.split(data)):
            Z = transform_minibatch(X, M, self.flank)
            n += sum(idx[2] for idx in index)
            yield index, Z
            if total > 0:
                callback(float(n)/total, 'transform')
    
