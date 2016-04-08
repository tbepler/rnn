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

import solvers

def null_func(*args, **kwargs):
    pass

class CharRNN(object):
    def __init__(self, n_in, n_out, layers, decoder=linear.Linear, itype='int32'
                 , solver=solvers.RMSprop(0.01)):
	self.n_out = n_out
	self.x = T.matrix(dtype=itype)
	self.y = T.matrix(dtype=itype)
	self.lens = T.vector(dtype='int32')
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

	yh, _ = theano.map(lambda i,n,ys: ys[n-1,i], [T.arange(b), self.lens], [y_layer])

        decode = decoder(m, n_out)
        self.weights.append(decode.weights)
        yh = decode(yh)

	self.yh = T.nnet.sigmoid(yh)
	self.loss_t = T.sum(T.nnet.binary_crossentropy(self.yh, y))
	self.tp = T.sum(self.yh > 0.5 and y)
	self.p = T.sum(y)
	self.tn = T.sum(self.yh <= 0.5 and not y)
	self.n = T.sum(not y)
	
        self.solver = solver
        #compile theano functions
        self._loss = theano.function([self.x, self.lens, self.y], [self.loss_t, self.tp, self.p, self.tn, self.n])

    def fit(self, data_train, validate=None, batch_size=256, max_iters=100, callback=null_func):
        steps = self.solver(BatchIter(data_train, batch_size, self.n_out), self.weights, [self.s, self.lens, self.y]
                            , self.loss_t, [self.tp, self.p, self.tn, self.n], max_iters=max_iters)
	loss_, tp_, p_, tn_, n_ = 0, 0, 0, 0, 0
        callback(0, 'fit')
        for it, (l,tp,p,tn,n) in steps:
	    loss_ += l
            tp_ += tp
	    p_ += p
	    tn_ += tn
	    n_ += n
            if it % 1 == 0:
                if validate is not None:
                    res = self.loss(validate, batch_size=batch_size, callback=callback)
                    res['TrainLoss'] = loss_/(p_+n_)
		    res['TrainSensitivity'] = float(tp_)/p_
		    res['TrainSpecificity'] = float(tn_)/n_
                    res['TrainAccuracy'] = float(tp_+tn_)/(p_+n_)
                else:
		    res = OrderedDict()
		    res['Loss'] = loss_/(p_+n_)
		    res['Sensitivity'] = float(tp_)/p_
		    res['Specificity'] = float(tn_)/n_
                    res['Accuracy'] = float(tp_+tn_)/(p_+n_)
		loss_, tp_, p_, tn_, n_ = 0, 0, 0, 0, 0
                yield res
            callback(it%1, 'fit')
            
    def loss(self, data, batch_size=256, callback=null_func):
        callback(0, 'loss')
	loss_, tp_, p_, tn_, n_ = 0, 0, 0, 0, 0
        for p,X,lens,Y in self.batch_iter(data, batch_size):
            l,tp,p,tn,n = self._loss(X, lens, Y)
	    loss_ += l
            tp_ += tp
	    p_ += p
	    tn_ += tn
	    n_ += n
            callback(p, 'loss')
	res = OrderedDict()
	res['Loss'] = loss_/(p_+n_)
	res['Sensitivity'] = float(tp_)/p_
	res['Specificity'] = float(tn_)/n_
	res['Accuracy'] = float(tp_+tn_)/(p_+n_)
        return res

    def batch_iter(self, data, size):
        for i in xrange(0, len(data), size):
            xs,ys = unzip(*data[i:i+size])
            n = len(xs)
            m = max(len(x) for x in xs)
            X = np.zeros((m,n), dtype=np.int32)
	    lens = np.array([len(x) for x in xs], dtype=np.int32)
	    Y = np.zeros((n,self.n_out), dtype=np.int32)
            for j in xrange(n):
                x = xs[j]
                k = len(x)
                X[:k,j] = x
		Y[j,ys[j]] = 1
            yield float(i+n)/len(data), X, lens, Y

class BatchIter(object):
    def __init__(self, data, size, n_out, shuffle=True):
        self.data = data
        self.size = size
	self.n_out = n_out
        self.shuffle = shuffle

    def __len__(self):
        return int(math.ceil(len(self.data)/float(self.size)))

    def __iter__(self):
        if self.shuffle:
	    random.shuffle(self.data)
	data = self.data
        size = self.size
        for i in xrange(0, len(data), size):
            xs,ys = unzip(*data[i:i+size])
            n = len(xs)
            m = max(len(x) for x in xs)
            X = np.zeros((m,n), dtype=np.int32)
	    lens = np.array([len(x) for x in xs], dtype=np.int32)
	    Y = np.zeros((n,self.n_out), dtype=np.int32)
            for j in xrange(n):
                x = xs[j]
                k = len(x)
                X[:k,j] = x
		Y[j,ys[j]] = 1
            yield float(i+n)/len(data), X, lens, Y
