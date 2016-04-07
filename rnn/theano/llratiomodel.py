import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict

import softmax
import crossent
import solvers

def null_func(*args, **kwargs):
    pass

class LLRatioModel(object):
    def __init__(self, models, itype='int32', solver=solvers.RMSprop(0.01)):
        self.x = T.matrix(dtype=itype)
        self.mask = T.matrix(dtype='int32')
        self.y = T.vector(dtype=itype)
        self.weights = []
        self.logprobs = []
        self.labels = []
        for label, model in models:
            yh = theano.clone(model.yh, {model.x: self.x[:-1], model.y: self.x[1:]})
            logprob = -T.sum(crossent.crossent(yh, self.x[1:])*self.mask[1:], axis=0)
            self.weights.extend(model.weights)
            self.logprobs.append(logprob)
            self.labels.append(label)
	self.logprobs = T.stack(self.logprobs, axis=1)
        self.yh = softmax.softmax(self.logprobs)
        self.loss_t = T.sum(crossent.crossent(self.yh, self.y))
        self.correct = T.sum(T.eq(T.argmax(self.yh, axis=1), self.y))
        self.count = self.y.size
        self.solver = solver
        #compile theano functions
        self._loss = theano.function([self.x, self.mask, self.y]
                                     , [self.loss_t, self.correct, self.count])

    def loss(self, data, batch_size=256, callback=null_func):
        callback(0, 'loss')
        loss_, correct_, count_ = 0, 0, 0
        for p, X, mask, Y in self.batch_iter(data, batch_size):
            l,c,n = self._loss(X, mask, Y)
            loss_ += l
            correct_ += c
            count_ += n
            callback(p, 'loss')
        return OrderedDict([('Loss', loss_/count_), ('Accuracy', float(correct_)/count_)])

    def batch_iter(self, data, size):
        for i in xrange(0, len(data), size):
            batch = data[i:i+size]
            xs,ys = zip(*batch)
            n = len(xs)
            m = max(len(x) for x in xs)
            X = np.zeros((m,n), dtype=xs[0].dtype)
            mask = np.ones((m,n), dtype=np.int32)
            for j in xrange(n):
                x = xs[j]
                k = len(x)
                X[:k,j] = x
                mask[k:,j] = 0
            Y = np.array(ys, dtype=np.int32)
            yield float(i+n)/len(data), X, mask, Y
        




