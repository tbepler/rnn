import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict
import math
import random
from Bio import SwissProt

import lstm
import linear
import softmax
import crossent
import crf

from rnn.minibatcher import BatchIter

import solvers

def null_func(*args, **kwargs):
    pass

class AnnoRNN(object):
    def __init__(self, n_in, units, layers, labels, decoder=linear.Linear, itype='int32'
                 , solver=solvers.RMSprop(0.01)):
       self.data = [T.matrix(dtype=itype), T.matrix(dtype=itype)]
       print self.data
       self.x = self.data[0].astype(itype) # T.matrix(dtype=itype)
       self.y = self.data[1].astype(itype) # T.matrix(dtype=itype)
       print self.x
       self.mask = T.matrix(dtype = "int8")
       self.weights = []
       k,b = self.x.shape
       y_layer = self.x
       self.y_layers = []
       layer = lstm.BLSTM(n_in, units)
       self.weights += [weight.astype(theano.config.floatX) for weight in layer.weights]
       y_layer = layer.scan(self.x, self.mask)
       self.y_layers.append(y_layer)
       self.yh = y_layer
       crf_layer = crf.CRF(units, labels, loss = crf.LikelihoodAccuracy())
       self.weights += [weight.astype(theano.config.floatX) for weight in crf_layer.weights]
       # self.yh = softmax.softmax(crf_layer)
       print self.yh
       loss, confusion = crf_layer.loss(self.yh, self.y)
       self.loss_t = T.sum(loss * T.shape_padright(self.mask))
       print self.loss_t
       self.count = T.sum(self.mask[1:])
       self.solver = solver
       #compile theano functions
       #self._loss = theano.function([self.data, self.mask], [self.loss_t, self.correct, self.count])
       #self._activations = theano.function([self.data], self.y_layers+[self.yh], givens={self.x:self.data})

       # self.data = T.matrix(dtype=itype)
       # self.x = self.data[:-1] # T.matrix(dtype=itype)
       # self.y = self.data[1:] # T.matrix(dtype=itype)
       # self.mask = T.matrix()
       # self.weights = []
       # k,b = self.x.shape
       # y_layer = self.x
       # self.y_layers = []
       # layer = lstm.BLSTM(n_in, units)
       # self.weights += [weight.astype(theano.config.floatX) for weight in layer.weights]
       # y0 = T.zeros((b, layers))
       # c0 = T.zeros((b, layers))
       # y_layer = layer.scan(y0, c0)
       # self.y_layers.append(y_layer)
       # decode = decoder(n_in, n_out)
       # self.weights += [weight.astype(theano.config.floatX) for weight in decode.weights]
       # yh = decode(y_layer)
       # self.yh = softmax.softmax(yh)
       # print self.y
       # self.loss_t = T.sum(crossent.crossent(self.yh, self.y)*self.mask[1:])
       # #self.correct = T.sum(T.eq(T.argmax(self.yh, axis=2), self.y)*self.mask[1:])
       # self.count = T.sum(self.mask[1:])
       # self.solver = solver
       # #compile theano functions
       # #self._loss = theano.function([self.data, self.mask], [self.loss_t, self.correct, self.count])
       # #self._activations = theano.function([self.data], self.y_layers+[self.yh], givens={self.x:self.data})
           


    def fit(self, data_train, validate=None, batch_size=256, max_iters=100, callback=null_func):
        steps = self.solver(BatchIter(data_train, batch_size), [self.x, self.y, self.mask], [self.loss_t], self.weights, max_iters=max_iters)
        #, [self.data, self.mask], self.loss_t, [self.correct, self.count], max_iters=max_iters)
        if validate is not None:
            validate = BatchIter(validate, batch_size, shuffle=False)
        train_loss, train_correct, train_n = 0, 0, 0
        callback(0, 'fit')
        for it, (l) in steps:
            print train_loss
            train_loss += l[0]
            #train_correct += c
            #train_n += n
            if it % 1 == 0:
                if validate is not None:
                    res = self.loss_iter(validate, callback=callback)
                    res['TrainLoss'] = train_loss/train_n
                    res['TrainAccuracy'] = float(train_correct)/train_n
                else:
                    res = OrderedDict([('Loss', train_loss/train_n)])
                    #                   , ('Accuracy', float(train_correct)/train_n)])
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

    def testing(self, data, batch_size = 20):
        testing_out = theano.function([self.x, self.y, self.mask], [self.loss_t])
        for batch in BatchIter(data, batch_size):
            print testing_out(batch)

def import_seq_data(filename):
    # Imports and preprocesses the sequence data
    # Takes file and returns sequence and labels of the sequence in np array format
    sequence_array = []
    label_array = []
    seq_dict = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19}
    max_len = 0
    for record in SwissProt.parse(open(filename)):
        if record.sequence_length > max_len:
            max_len = record.sequence_length
        labels = [0]*record.sequence_length
        for feature in record.features:
            if feature[0] == "HELIX":
                new_label = 1
            elif feature[0] == "STRAND":
                new_label = 2
            elif feature[0] == "TURN":
                new_label = 3
            else:
                continue
            begin = feature[1]
            end = feature[2]
            for i in range(begin-1, end-1):
                labels[i] = new_label
        label_array.append(labels) 
        encoded_seq = []
        for char in record.sequence:
            encoded_seq += [seq_dict[char]]
        sequence_array.append(encoded_seq)
    return sequence_array, label_array, max_len

if __name__ == '__main__':
    sequences, ss_labels, length = import_seq_data("uniprot_short.txt")
    print length
    samples = 20
    labels = 4
    model = AnnoRNN(20, 5, 6, labels)
    # data = np.random.randint(0, labels-1, (length, samples)).astype(np.int32)
    # labeled_data = data
    # print data
    data = np.array([sequences, ss_labels])
    #print data
    #iterator = model.batch_iter(data, 64)
    #for i in iterator:
    #    model.fit(i)
    fit_data = model.fit(data, batch_size = 10)
    for i in fit_data:
        print i
    print "Begin testing"
    test_data = model.testing(data, batch_size = 19)
    print test_data
