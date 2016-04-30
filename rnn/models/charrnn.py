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

class CharRNNGraph(object):
    def __init__(self, n_in, n_out, layers, decoder=linear.Linear, weights=None):
        self.layers = []
        m = n_in
        for n in layers:
            L = lstm.LSTM(m, n)
            self.layers.append(L)
            m = n
        self.decoder = decoder(m, n_out) 
        if weights is not None:
            self.weights = weights

    @property
    def nlayers(self):
        return len(self.layers)

    @property
    def weights(self):
        return [layer.weights for weights in self.layers] + [self.decoder.weights]

    @weights.setter
    def weights(self, ws):
        for i in xrange(len(self.layers)):
            self.layers[i].weights.set_value(weights[i])
        self.decoder.weights.set_value(weights[-1])

    def copy_target(self, target):
        from copy import copy
        layers = [copy(l) for l in self.layers]
        for layer in layers:
            layer.weights = theano.shared(layer.weights.get_value(), target=target)
        decoder = copy(self.decoder)
        decoder.weights = theano.shared(decoder.weights.get_value(), target=target)
        cpy = copy(self)
        cpy.layers = layers
        cpy.decoder = decoder
        return cpy

    def __call__(self, X, y0=None, c0=None):
        k,b = X.shape
        yh = X
        y_layers = []
        cs = []
        for i in xrange(len(self.layers)):
            layer = self.layers[i]
            if y0 is not None:
                y0i = y0s[i]
            else:
                y0i = T.zeros((b,layer.units))
            if c0 is not None:
                c0i = c0s[i]
            else:
                c0i = T.zeros((b, layer.units))
            yh, c = layer.scanl(y0i, c0i, yh, truncate_gradient=truncate_gradient)
            y_layers.append(yh)
            cs.append(c)
        yh = softmax.softmax(self.decoder(yh))
        return yh, y_layers, cs
    
class CharRNN_new(object):
    def __init__(self, n_in, n_out, layers, decoder=linear.Linear, solver=solvers.RMSprop(0.01)):
        self.n_in = n_in
        self.n_out = n_out
        self.layers = layers
        self.decoder = decoder
        self.solver = solver
        self.setup()

    def setup(self, weights=None):
        self._theano_model = CharRNNGraph(self.n_in, self.n_out, self.layers, decoder=self.decoder
                , weights=weights)

    @property
    def weights(self):
        return [w.get_value(borrow=True) for w in self._theano_model.weights]

    @property
    def parameters(self):
        params = {}
        params['n_in'] = self.n_in
        params['n_out'] = self.n_out
        params['layers'] = self.layers
        params['decoder'] = self.decoder
        params['solver'] = self.solver
        return params

    def __getstate__(self):
        state = {}
        state['weights'] = self.weights
        state['params'] = self.parameters
        return state

    def __setstate__(self, state):
        for k,v in state['params'].iteritems():
            setattr(self, k, v)
        self.setup(weights=state['weights'])

    def _theano_crossent(self, Yh, Y, mask):
        return crossent.crossent(Yh, Y)*mask

    def _theano_confusion(self, Yh, Y, mask):
        Yh = T.argmax(Yh, axis=-1)
        shape = list(Yh.shape) + [self.n_out, self.n_out]
        C = T.zeros(shape)
        i,j = T.mgrid[0:C.shape[0], 0:C.shape[1]]
        C = T.set_subtensor(C[i,j,Y,Yh], 1)
        mask = mask.dimshuffle(*(mask.shape + ['x','x']))
        C = C*mask
        return C

    def _theano_loss(self, X, Y, mask, models, axis=0):
        idxs = [T.arange(i, X.shape[1], len(models)) for i in len(models)]
        Ys = [Y[:,idx] for idx in idxs]
        masks = [mask[:,idx] for idx in idxs]
        Yhs = [model(X[:,idx])[0] for model,idx in zip(models, idxs)]
        cents = [T.sum(self._theano_crossent(Yh, Yi, maski), axis=axis) for Yh,Yi,maski in zip(Yhs, Ys, masks)]
        confs = [T.sum(self._theano_confusion(Yh, Yi, maski), axis=axis) for Yh,Yi,maski in zip(Yhs, Ys, masks)]
        if axis == 0:
            idxs = T.concatenate(idxs, axis=0)
            idx_inv = T.arange(X.shape[1])[idxs]
            cent = T.concatenate(cents, axis=0)[idx_inv]
            conf = T.concatenate(confs, axis=0)[idx_inv]
        else:
            cent = sum(cents)
            conf = sum(confs)
        return cent, conf

    def _compile_loss(self, dtype='int32', devices=[], axis=0):
        if len(devices) == 0:
            devices.append(self._theano_model)
        else:
            devices = [self._theano_model.copy_target(target) for target in devices]
        data = T.matrix(dtype=dtype)
        X = data[:-1]
        Y = data[1:]
        mask = T.matrix(dtype='float32')
        results = self._theano_loss(data[:-1], data[1:], mask[1:], devices, axis=axis)
        f = theano.function([data, mask], results)
        return f

    def loss_iter(self, data, callback=null_func, devices=[]):
        f = self._compile_loss(dtype=data.dtype, devices=devices, axis=0)
        callback(0, 'loss')
        i = 0
        for X, mask in data:
            res = f(X, mask)
            i += 1
            p = float(i)/len(data)
            callback(p, 'loss')
            yield res

    def loss(self, data, callback=null_func, devices=[], axis=None):
        if axis == 0:
            return self.loss_iter(data, callback=callback, devices=devices)
        f = self.compile_loss(dtype=data.dtype, devices=devices, axis=axis)
        callback(0, 'loss')
        cent, conf, i = 0, 0, 0
        for X, mask in data:
            cent_, conf_ = f(X, mask)
            cent += cent_
            conf += conf_
            i += 1
            p = float(i)/len(data)
            callback(p, 'loss')
        return cent, conf

    def _theano_gradient_truncated(self, X, Y, mask, model, backprop):
        n = model.nlayers
        def step(i, *args):
            y0 = args[:n]
            c0 = args[n:2*n]
            X = args[2*n]
            Y = args[2*n+1]
            mask = args[2*n+2]
            Yh,y_layers,cs = model(X[i:i+backprop], y0=y0, c0=c0)
            loss = T.sum(self._theano_crossent(Yh, Y[i:i+backprop], mask[i:i+backprop]))
            gw = theano.grad(loss, model.weights)
            ys = [y_layer[-1] for y_layer in y_layers]
            return gw + ys + cs
            

        pass

    def _theano_gradient(self, X, Y, model, truncated_backprop=0):
        pass



class CharRNN(object):
    def __init__(self, n_in, n_out, layers, decoder=linear.Linear, itype='int32'
                 , solver=solvers.RMSprop(0.01)):
        self.data = T.matrix(dtype=itype)
        self.x = self.data[:-1] # T.matrix(dtype=itype)
        self.y = self.data[1:] # T.matrix(dtype=itype)
        self.mask = T.matrix(dtype='int32')
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

    def graph_predict(self, X, device=None):
        k,b = self.x.shape
        yh = X
        for layer in self.layers:
            n = layer.units
            y0 = T.zeros((b, n))
            c0 = T.zeros((b, n))
            yh = layer.scanl(y0, c0, yh)
        yh = softmax.softmax(self.decoder(yh))
        if device is not None:
            weights = [layer.weights for layer in self.layers] + [self.decoder.weights]
            subst = {w: w.transfer(device) for w in weights}
            yh = theano.clone(yh, subst)
        return yh
        
    def graph_loss(self, Yh, Y, mask):
        return T.sum(crossent.crossent(Yh, Y)*mask)

    def graph_correct(self, Yh, Y, mask):
        return T.sum(T.eq(T.argmax(Yh, axis=1), Y)*mask)

    def graph_fit(self, data, mask, devices=None):
        X = data[:-1]
        Y = data[1:]
        mask = mask[1:]
        if devices is not None:
            k,b = data.shape
            loss, correct = 0, 0
            for i in xrange(len(devices)):
                idx = T.arange(i, b, len(devices))
                yh = self.graph_predict(X[:,idx], device=devices[i])
                loss += self.graph_loss(yh, Y[:,idx], mask[:,idx])
                correct += self.graph_correct(yh, Y[:,idx], mask[:,idx])
        else:
            yh = self.graph_predict(X)
            loss = self.graph_loss(yh, Y, mask)
            correct = self.graph_correct(yh, Y, mask)
        count = T.sum(self.mask)
        return loss, correct, count


    def fit(self, data_train, validate=None, batch_size=256, max_iters=100, callback=null_func):
        weights = [layer.weights for layer in self.layers] + [self.decoder.weights]
        data_symbol = T.matrix(dtype=data_train.dtype)
        mask_symbol = T.matrix(dtype='int32')
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
            mask = np.ones((m,n), dtype=np.int32)
            for j in xrange(n):
                x = xs[j]
                k = len(x)
                X[:k,j] = x
                mask[k:,j] = 0
            yield float(i+n)/len(data), X, mask


