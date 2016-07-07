import theano
import theano.tensor as T
import numpy as np
np.set_printoptions(threshold=np.nan)
from collections import OrderedDict
import math
from operator import add
import random
from Bio import SwissProt
import collections
import sys
from datetime import datetime

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
    def __init__(self, n_in, units, layers, labels, loss_scaler, decoder=linear.Linear, lambdal1=0.001, itype='int32', solver=solvers.RMSprop(0.1, decay=solvers.GeomDecay(0.999))):
        self.data = [T.matrix(dtype=itype), T.matrix(dtype=itype)]
        self.x = self.data[0].astype(itype) # T.matrix(dtype=itype)
        self.y = self.data[1].astype(itype) # T.matrix(dtype=itype)
        self.mask = T.matrix(dtype = "int8")
        # Weights of each layers
        self.weights = []
        # Location of the weights of the layers in the weight array (needed to find the historical data to update in the solver)
        self.layerloc = []
        k,b = self.x.shape
        y_layer = self.x
        #self.y_layers = []
        lstm_layer = lstm.DiffBLSTM(n_in, units)
        start = len(self.weights)
        self.weights += [weight for weight in lstm_layer.weights]
        end = len(self.weights)
        self.layerloc += [[start, end]]
        y_layer, c = lstm_layer.scan(self.x, self.mask)
        #self.y_layers.append(y_layer)
        self.yh = y_layer
        crf_layer = crf.CRF(units, labels, loss = crf.LikelihoodAccuracy())
        start = len(self.weights)
        self.weights += [weight for weight in crf_layer.weights]
        end = len(self.weights)
        self.layerloc += [[start, end]]
        # self.yh = softmax.softmax(crf_layer)
        loss, confusion, lastyh, self.yy= crf_layer.loss(self.yh, self.y)
        l1 = T.sum(abs(lstm_layer.multiplier))
        self.lastyh = lastyh
        self.count = T.sum(self.mask)
        self.loss_t = (T.sum(loss * T.shape_padright(self.mask) * loss_scaler))/self.count + l1*lambdal1
        self.confusion = confusion * T.shape_padright(T.shape_padright(self.mask))
        self.solver = solver
        # The layer that is differentiable and the one after it (needed to update weights and historical info in solver)
        self.difflayers = [lstm_layer, crf_layer]
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
       # self.correct = T.sum(T.eq(T.argmax(self.yh, axis=2), self.y)*self.mask)
       # self.solver = solver
       # #compile theano functions
       # #self._loss = theano.function([self.data, self.mask], [self.loss_t, self.correct, self.count])
       # #self._activations = theano.function([self.data], self.y_layers+[self.yh], givens={self.x:self.data})
           


    def fit(self, data_train, validate=None, batch_size=256, max_iters=50, callback=null_func):
        steps = self.solver(BatchIter(data_train, batch_size), [self.x, self.y, self.mask, self.difflayers, self.layerloc], [self.loss_t, self.confusion, self.count], self.weights, max_iters=max_iters)
        #, [self.data, self.mask], self.loss_t, [self.correct, self.count], max_iters=max_iters)
        if validate is not None:
            validate = BatchIter(validate, batch_size, shuffle=False)
        train_loss, train_correct, train_n = 0, 0, 0
        callback(0, 'fit')
        for it, (l, confusion, n) in steps:
            #print train_loss
            #print l
            c = np.sum([np.trace(square) for row in confusion for square in row])
            #print n
            train_loss += l
            train_correct += c
            train_n += n
            if it % 1 == 0:
                if validate is not None:
                    res = self.loss_iter(validate, callback=callback)
                    res['TrainLoss'] = train_loss/train_n
                    res['TrainAccuracy'] = float(train_correct)/train_n
                else:
                    res = OrderedDict([('Loss', train_loss/train_n), ('Accuracy', float(train_correct)/train_n)])
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
        testing_out = theano.function([self.x, self.y, self.mask], [self.loss_t, self.confusion, self.count, self.lastyh, self.yy])
	train_correct = 0
	train_n = 0
        for batch in BatchIter(data, batch_size):
            loss, confusion, count, yh, yy =  testing_out(batch[0], batch[1], batch[2])
            correct = np.sum([np.trace(square) for row in confusion for square in row])
            train_correct += correct
            train_n += count
            #print loss
            #print batch[1]
            #print confusion
            print np.array(T.argmax(yh, axis = 2).eval())
            print yy
	print "Batch accuracy:"
	print float(train_correct)/train_n

def import_seq_data(filename):
    # Imports and preprocesses the sequence data
    # Takes file and returns sequence and labels of the sequence in np array format
    sequence_array = []
    label_array = []
    seq_dict = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19}
    label_dict = {"HELIX": 1, "STRAND": 2, "TURN":3}
    num_labels = len(label_dict) + 1
    max_len = 0
    final_frequencies = [0] * num_labels
    # gets next record from the parsed file from SwissProt
    for record in SwissProt.parse(open(filename)):
        labels = [0]*record.sequence_length
        frequencies = []
        changed = False
        for feature in record.features:
            # for each feature, if it is in the label dictionary, add corresponding label to array and update label frequencies
            if feature[0] in label_dict:
                new_label = label_dict[feature[0]]
            else:
                continue
            begin = feature[1]
            end = feature[2]
            # there is a label in the sequence and adds these labels to the full array
            changed = True
            for i in range(begin-1, end-1):
                labels[i] = new_label
        encoded_seq = []
        # translates the AA sequence into digits
        for char in record.sequence:
            if char in seq_dict:
                encoded_seq += [seq_dict[char]]
            else:
                # if cannot find the AA, then ignore label and sequence
                changed = False
        if not changed:
            continue
        counter = dict(collections.Counter(labels))
        frequencies = [counter[i] if i in counter else 0 for i in range(num_labels)]
        final_frequencies = map(add, final_frequencies, frequencies)
        label_array.append(labels) 
        sequence_array.append(encoded_seq)
        if record.sequence_length > max_len:
            max_len = record.sequence_length
    return sequence_array, label_array, num_labels, final_frequencies

if __name__ == '__main__':
    currentTime = datetime.now() 
    startTime = currentTime
    #orig_stdout = sys.stdout
    #f = file('outnew.txt', 'w')
    #sys.stdout = f

    sequences, ss_labels, labels, label_frequencies = import_seq_data("uniprot_human.txt")
    a = np.array(ss_labels)
    #print 1-float(sum([c>0 for b in a for c in b]))/sum([len(b)for b in a])
    samples = 20
    train_split = 0.7
    layers = 2
    units = 10
    label_frequencies = [math.sqrt(l) for l in label_frequencies]
    total_labels = sum(label_frequencies)
    #print label_frequencies
    loss_scaler = [0 if freq == 0 else float(total_labels)/freq for freq in label_frequencies]
    #print loss_scaler
    model = AnnoRNN(samples, units, layers, labels, loss_scaler)
    # data = np.random.randint(0, labels-1, (length, samples)).astype(np.int32)
    # labeled_data = data
    # print data
    data = np.array([sequences, ss_labels])[:, :10]
    print data.shape
    print "Inputs: %d" % len(data[0])
    print "Layers: %d" % layers
    print "Units: %d" % units
    #print data.shape
    train_data, test_data = np.hsplit(data, [int(train_split*data.shape[1])])
    #print train_data.shape
    #print test_data.shape
    #print data
    #iterator = model.batch_iter(data, 64)
    #for i in iterator:
    #    model.fit(i)
    print "Data preprocessing time: %s" % (datetime.now() - currentTime)
    currentTime = datetime.now()

    fit_data = model.fit(train_data, batch_size = 50)

    print "Model construction time: %s" % (datetime.now() - currentTime)
    currentTime = datetime.now()

    count = 0
    print "Begin training"
    for i in fit_data:
	count += 1
	print "Epoch %d" % count
        print i
        print "Training time: %s" % (datetime.now() - currentTime)
        currentTime = datetime.now()

    print "Begin testing"
    test_data = model.testing(test_data, batch_size = 50)
    print test_data
    print "Testing time: %s" % (datetime.now() - currentTime)
    currentTime = datetime.now()
    print "Total runtime: %s" % (datetime.now() - startTime) 

    #sys.stdout = orig_stdout
    #f.close()
