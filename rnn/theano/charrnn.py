import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict
import math
import random

import lstm
import linear
import softmax
import crossent

from rnn.minibatcher import BatchIter

import solvers

def null_func(*args, **kwargs):
    pass

class CharRNN(object):
    def __init__(self, n_in, n_out, layers, decoder=linear.Linear, itype='int32'
                 , solver=solvers.RMSprop(0.01)):
        self.data = T.matrix(dtype=itype)
        self.x = self.data[:-1] # T.matrix(dtype=itype)
        self.y = self.data[1:] # T.matrix(dtype=itype)
        self.mask = T.matrix()
        self.weights = []
        k,b = self.x.shape
        y_layer = self.x
        self.y_layers = []
        m = n_in
        for n in layers:
            layer = lstm.LSTM(m, n)
            self.weights.append(layer.weights)
            y0 = T.zeros((b, n))
            c0 = T.zeros((b, n))
            y_layer, _ = layer.scanl(y0, c0, y_layer)
            self.y_layers.append(y_layer)
            m = n
        decode = decoder(m, n_out)
        self.weights.append(decode.weights)
        yh = decode(y_layer)
        self.yh = softmax.softmax(yh)
        self.loss_t = T.sum(crossent.crossent(self.yh, self.y)*self.mask[1:])
        self.correct = T.sum(T.eq(T.argmax(self.yh, axis=2), self.y)*self.mask[1:])
        self.count = T.sum(self.mask[1:])
        self.solver = solver
        #compile theano functions
        self._loss = theano.function([self.data, self.mask], [self.loss_t, self.correct, self.count])
        self._activations = theano.function([self.data], self.y_layers+[self.yh], givens={self.x:self.data})

    def fit(self, data_train, validate=None, batch_size=256, max_iters=100, callback=null_func):
        steps = self.solver(BatchIter(data_train, batch_size), self.weights, [self.data, self.mask]
                            , self.loss_t, [self.correct, self.count], max_iters=max_iters)
        if validate is not None:
            validate = BatchIter(validate, batch_size, shuffle=False)
        train_loss, train_correct, train_n = 0, 0, 0
        callback(0, 'fit')
        for it, (l,c,n) in steps:
            train_loss += l
            train_correct += c
            train_n += n
            if it % 1 == 0:
                if validate is not None:
                    res = self.loss_iter(validate, callback=callback)
                    res['TrainLoss'] = train_loss/train_n
                    res['TrainAccuracy'] = float(train_correct)/train_n
                else:
                    res = OrderedDict([('Loss', train_loss/train_n)
                                       , ('Accuracy', float(train_correct)/train_n)])
                train_loss, train_correct, train_n = 0, 0, 0
                yield res
            callback(it%1, 'fit')
            
    def loss_iter(self, data, callback=null_func):
        callback(0, 'loss')
        loss_, correct_, count_ = 0, 0, 0
        i = 0
        for X,mask in data:
            i += 1
            p = float(i)/len(data)
            l,c,n = self._loss(X, mask)
            loss_ += l
            correct_ += c
            count_ += n
            callback(p, 'loss')
        return OrderedDict([('Loss', loss_/count_), ('Accuracy', float(correct_)/count_)])

    def loss(self, data, batch_size=256, callback=null_func):
        iterator = BatchIter(data, batch_size)
        return self.loss_iter(iterator, callback=callback)

    def activations(self, data, batch_size=256, callback=null_func):
        callback(0, 'activations')
        for p,X,lens in self.batch_iter_no_mask(data, batch_size):
            acts = self._activations(X)
            for i in xrange(len(lens)):
                n = lens[i]
                yield [act[:n,i,:] for act in acts]

    def batch_iter_no_mask(self, data, size):
        for i in xrange(0, len(data), size):
            xs = data[i:i+size]
            n = len(xs)
            m = max(len(x) for x in xs)
            X = np.zeros((m,n), dtype=np.int32)
            for j in xrange(n):
                x = xs[j]
                k = len(x)
                X[:k,j] = x
            yield float(i+n)/len(data), X, [len(x) for x in xs]

    def batch_iter(self, data, size):
        for i in xrange(0, len(data), size):
            xs = data[i:i+size]
            n = len(xs)
            m = max(len(x) for x in xs)
            X = np.zeros((m,n), dtype=np.int32)
            mask = np.ones((m,n), dtype=np.float32)
            for j in xrange(n):
                x = xs[j]
                k = len(x)
                X[:k,j] = x
                mask[k:,j] = 0
            yield float(i+n)/len(data), X, mask


