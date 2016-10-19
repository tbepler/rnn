from get_unused_gpus import get_unused_gpus
gpu = get_unused_gpus()
print "gpu%s" % gpu[0]
import os
os.environ["THEANO_FLAGS"] = "floatX=float32,device=gpu" + str(gpu[0]) + ",gcc.cxxflags='-march=core2',lib.cnmem=0.9,scan.allow_gc=True,base_compiledir='/tmp/rkc10/.theano'"
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
import itertools

from minibatcher import BatchIter

import solvers


def null_func(*args, **kwargs):
    pass

class AnnoRNN(object):
    def __init__(self, lstm_type, n_in, units, labels, loss, loss_scaler, sections=4, decoder=linear.Linear, lambdal1=0.00001, lambdal2=0, lambdaprll=0.01, label_sects = 1, itype='int32', learning_rate = 0.001, decay = 0.9, mint=False, batch_norm=False):
        print "Sections: %s" % sections
        print "Learning rate: %s" % learning_rate
        print "Decay: %s" % decay
        solver = solvers.RMSprop(learning_rate, decay=solvers.GeomDecay(decay))
        if label_sects == 1:
            self.data = [T.matrix(dtype=itype), T.matrix(dtype=itype)]
        else:
            self.data = [T.matrix(dtype=itype), T.tensor3(dtype=itype)]
        self.samples = n_in
        self.labels = labels
        self.x = self.data[0].astype(itype) # T.matrix(dtype=itype)
        self.y = self.data[1].astype(itype) # T.matrix(dtype=itype)
        self.mask = T.matrix(dtype = "int8")
        # Weights of each layers
        self.weights = []
        # Location of the weights of the layers in the weight array (needed to find the historical data to update in the solver)
        self.layerloc = []
        k,b = self.x.shape
        y_layer = self.x
        self.label_sects = label_sects
        #self.y_layers = []
        
        # Initializes the LSTM Layer
        if lstm_type == "ParallelLSTM":
            lstm_layer = getattr(lstm, lstm_type)(n_in, units[0], sections, batch_norm=batch_norm, mint=mint)
        elif lstm_type not in ("DiffLayeredBLSTM", "LayeredBLSTM", "LayeredLSTM"):
            lstm_layer = getattr(lstm, lstm_type)(n_in, units[0], batch_norm=batch_norm, mint=mint)
        else:
            lstm_layer = getattr(lstm, lstm_type)(n_in, units, batch_norm=batch_norm, mint=mint)
        start = len(self.weights)
        print lstm_layer
        #self.weights += [weight for weights in lstm_layer.weights for weight in weights]
        self.weights += lstm_layer.weights
        if lambdal2 != None:
            l2 = lstm_layer.l2
        else:
            l2 = 0
        print self.weights
        end = len(self.weights)
        self.layerloc += [[start, end]]
        if lstm_type in ("LSTM", "DiffLSTM", "ParallelLSTM"):
            y_layer, _ = lstm_layer.scanl(self.x, mask=self.mask)
        else:
            y_layer = lstm_layer.scan(self.x, mask=self.mask)


        # For parallel layers, calculate additional cost function for dot product of outputs for each section
        if lstm_type == "ParallelLSTM":
            y_sections = []
            secunits = lstm_layer.secunits
            for i in range(lstm_layer.sections):
                y_sections += [y_layer[:,:,i*secunits:(i+1)*secunits]]
            parallel_loss = 0
            for pair in itertools.combinations(y_sections,2):
                parallel_loss += T.sum(abs(T.tensordot(pair[0], pair[1], axes = [[0,1,2],[0,1,2]])))
        else:
            parallel_loss = 0

        #self.y_layers.append(y_layer)
        self.yh = y_layer
        if lstm_type == "ParallelLSTM":
            crf_layer = crf.CRF(units[-1]*sections, labels, loss = getattr(crf, loss)())
        else:
            if label_sects == 1:
                crf_layer = crf.CRF(units[-1], labels, loss = getattr(crf, loss)())
            else:
                crf_layer = crf.MultiCRF(units[-1], labels, loss = getattr(crf, loss)(), label_sects=self.label_sects)
        start = len(self.weights)
        self.weights += crf_layer.weights
        print self.weights
        end = len(self.weights)
        self.layerloc += [[start, end]]
        # self.yh = softmax.softmax(crf_layer)
        loss, confusion, lastyh, self.yy= crf_layer.loss(self.yh, self.y)
        if lstm_type in ("DiffLSTM", "DiffBLSTM", "DiffLayeredBLSTM"):
            print "its here"
            l1 = T.sum(abs(lstm_layer.multiplier))
        else:
            l1 = 0
        self.confusion = confusion * T.shape_padright(T.shape_padright(self.mask))
        self.lastyh = lastyh
        self.count = T.sum(self.mask) * self.label_sects
        if loss_scaler == None:
            loss_scaler = 1
        print "Lambda parallel: %s" % lambdaprll
        print "Lambda l2: %s" % lambdal2
        print "Lambda l1: %s" % lambdal1
        #crf_loss = T.sum(crf_layer.weights[1])
        crf_loss = 0
        self.loss_t = (T.sum(loss * T.shape_padright(self.mask) * loss_scaler))/self.count + l1*lambdal1 + l2*lambdal2 + parallel_loss*lambdaprll + crf_loss

        if label_sects == 1:
            self.correct = T.sum(T.eq(T.argmax(self.yh, axis=2), self.y)*self.mask)
        else:
            self.correct = T.sum(T.eq(T.argmax(self.yh, axis=-1), self.y)*T.shape_padaxis(self.mask,1))
            #(confusion * T.shape_padaxis(self.mask, 1)).sum()
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
           


    def fit(self, data_train, validate=None, batch_size=256, max_iters=20, callback=null_func):
        if self.label_sects == 1:
            single_out = True
        else:
            single_out = False
        steps = self.solver(BatchIter(data_train, batch_size, single_out), [self.x, self.y, self.mask, self.difflayers, self.layerloc], [self.loss_t, self.confusion, self.count], self.weights, max_iters=max_iters)
        #, [self.data, self.mask], self.loss_t, [self.correct, self.count], max_iters=max_iters)
        if validate is not None:
            validate = BatchIter(validate, batch_size, single_out, shuffle=False)
        train_loss, train_correct, train_n = 0, 0, 0
        callback(0, 'fit')
        for it, (l, c, n) in steps:
            #print train_loss
            #print l
            c = np.sum([np.trace(square) for row in c for square in row])
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
        if self.label_sects == 1:
            single_out = True
        else:
            single_out = False
        iterator = BatchIter(data, batch_size, single_out)
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

    def testing(self, data, batch_size = 30):
        testing_out = theano.function([self.x, self.y, self.mask], [self.loss_t, self.confusion, self.count, self.lastyh, self.yy, self.x])
        testing_correct = 0
        testing_n = 0
        total_loss = 0
        if self.label_sects == 1:
            single_out = True
        else:
            single_out = False
        total_sample_dict = {i:0 for i in range(self.samples)}
        total_label_dict = {i:0 for i in range(self.labels)}
        correct_sample_dict = {i:0 for i in range(self.samples)}
        correct_label_dict = {i:0 for i in range(self.labels)}
        for batch in BatchIter(data, batch_size, single_out):
            loss, confusion, count, yh, yy, x =  testing_out(batch[0], batch[1], batch[2])
            #print "Confusion: %s" % confusion
            correct_array = np.array([[np.trace(row) for row in square] for square in confusion])
            mask = np.array([[np.sum(row) for row in square] for square in confusion])
            total_confusion = np.sum(confusion, axis=(0,1))
            #print correct_array
            for i in range(correct_array.shape[0]):
                for j in range(correct_array.shape[1]):
                    if mask[i][j] == 1:
                        samp = x[i][j]
                        lb = yy[i][j]
                        total_sample_dict[samp] += 1
                        total_label_dict[lb] += 1
                        if correct_array[i][j] == 1:
                            correct_sample_dict[samp] += 1
                            correct_label_dict[lb] += 1
            testing_correct += np.sum(correct_array)
            testing_n += count
            total_loss += loss
            #print loss
            #print batch[1]
            #print confusion
            #print np.array(T.argmax(yh, axis = -1).eval())
            #print np.array(T.argmax(yh, axis = -1).eval()).shape
            #print yy
            #print yy.shape
        sample_acc = [0]*self.samples
        label_acc = [0]*self.labels
        print total_sample_dict
        print correct_sample_dict
        print total_label_dict
        print correct_label_dict
        for key, value in total_sample_dict.iteritems():
            if key in correct_sample_dict and value != 0:
                sample_acc[key] = float(correct_sample_dict[key])/value
        for key, value in total_label_dict.iteritems():
            if key in correct_label_dict and value != 0:
                label_acc[key] = float(correct_label_dict[key])/value
        total_acc = float(testing_correct)/testing_n
        print "Batch accuracy: %s" % total_acc
        print "Input accuracy breakdown: %s" % sample_acc
        print "Label accuracy breakdown: %s" % label_acc
        print "Total confusion matrix: %s" % total_confusion
        return [total_loss, total_acc, sample_acc, label_acc]

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
    import sys, getopt

    argv = sys.argv[1:]
    lstm_type = 'BLSTM'
    loss = 'LikelihoodAccuracy'
    epochs = 10
    batch_size = 50
    units = [20]
    mint = False
    database = 0
    bnorm = False
    scaler = False
    try:
        opts, args = getopt.getopt(argv,"hl:e:b:u:ms:nco:",["lstmtype=","epochs=", "batchsize=", "units=", "database=", "loss="])
    except getopt.GetoptError:
        print 'test.py -l <lstmtype> -e <epochs> -b <batchsize>'
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print 'test.py -l <lstmtype> -e <epochs> -b <batchsize> -u <units> -m <multiplicativeintegration> -db <database>, -n <normalizebatch> -c <scaler> -o <loss>'
            sys.exit()
        elif opt in ("-l", "--lstmtype"):
            # Type of lstm to run (LSTM, LayeredLSTM, BLSTM, LayeredBLSTM, DiffLSTM, DiffBLSTM, DiffLayeredBLSTM, ParallelLSTM)
            lstm_type = arg
        elif opt in ("-e", "--epochs"):
            # Number of epochs
            epochs = int(arg)
        elif opt in ("-b", "--batchsize"):
            # Number of imputs per batch
            batch_size = int(arg)
        elif opt in ("-u", "--units"):
            # Number of units for the LSTM layer
            units = map(int, arg.split(","))
        elif opt in ("-m", "--multiplicativeintegration"):
            # Uses multiplicative integration instead of additive integration for lstm
            mint = True
        elif opt in ("-s", "--database"):
            # If 0, runs only a few samples (for code testing purposes only)
            # If 1, runs entire pdb/swissprot db
            # If 2, runs spider db
            database = int(arg)
        elif opt in ("-n", "--normalizebatch"):
            # Runs batch normalization between LSTM and decoder
            bnorm = True
        elif opt in ("-c", "--scaler"):
            # Scales the loss to the proportion of labels
            scaler = True
        elif opt in ("o", "--loss"):
            loss = arg
        
    print 'LSTM type is: %s' % lstm_type
    print 'Number of Epochs is: %s' % epochs
    print 'Batch size is: %s' % batch_size
    print "Units: %s" % units
    print "Multiplicative Inverse? %s" % mint
    print "Database: %s" % database
    print "Batch normalize? %s" % bnorm 
    print "Scaler? %s" % scaler
    print "Loss: %s" % loss


    currentTime = datetime.now() 
    startTime = currentTime
    #orig_stdout = sys.stdout
    #f = file('NoDiffBaNormPCE.txt', 'w')
    #sys.stdout = f

    if database == 0:
        #sequences, ss_labels, labels, label_frequencies = import_seq_data("/n/scratch2/rkc10/uniprot_short.txt")
        labels = 5
        samples = 5
        label_sects = 1
        train_split = 0.7
        sequences = [[random.randint(0,labels-1) for i in range(100)] for i in range(50)]
        ss_labels = [[0,0] + [seq[i-1] for i in range(len(seq)-2)] for seq in sequences]
        ss_labels = sequences
        scaler = False

    elif database == 1:
        sequences, ss_labels, labels, label_frequencies = np.load('/n/scratch2/rkc10/finaldata.npy') 
        samples = 20
        labels = 10
        label_sects = 10
        train_split = 0.7
        print label_frequencies
        label_frequencies = [math.sqrt(l) for l in label_frequencies]
        total_labels = sum(label_frequencies)

    else:
        sequences, ss_labels, labels, label_frequencies = np.load('/n/scratch2/rkc10/train_predata.npy') 
        test_sequences, test_ss_labels, _, _ = np.load('/n/scratch2/rkc10/test_predata.npy')
        samples = 20
        labels = 4
        label_sects = 1
        print label_frequencies
        label_frequencies = [math.sqrt(l) for l in label_frequencies]
        total_labels = sum(label_frequencies)

    if scaler:
        loss_scaler = [0 if freq == 0 else float(total_labels)/freq for freq in label_frequencies]
    else:
        loss_scaler = None
    #print 1-float(sum([c>0 for b in a for c in b]))/sum([len(b)for b in a])
    model = AnnoRNN(lstm_type, samples, units, labels, loss, loss_scaler, mint=mint, batch_norm=bnorm, label_sects=label_sects)
    # data = np.random.randint(0, labels-1, (length, samples)).astype(np.int32)
    # labeled_data = data
    # print data
    if database == 0:
        data = np.array([sequences, ss_labels])
        print data.shape
        #print data.shape
        train_data, test_data = np.hsplit(data, [int(train_split*data.shape[1])])
        print train_data.shape
        print test_data.shape
        #print data
    elif database == 1:
        sequences = np.array(sequences)
        print sequences.shape
        ss_labels = np.array(ss_labels)
        print ss_labels.shape
        split_size = int(len(sequences)*train_split)
        train_data = [sequences[:split_size], ss_labels[:split_size]]
        test_data = [sequences[split_size:], ss_labels[split_size:]]
    else:
        train_data = np.array([sequences, ss_labels])
        test_data = np.array([test_sequences, test_ss_labels])
        print train_data.shape
        print test_data.shape
    print "Training Inputs: %d" % len(train_data[0])
    print "Testing Inputs: %d" % len(test_data[0])
    print "Data preprocessing time: %s" % (datetime.now() - currentTime)
    currentTime = datetime.now()

    fit_data = model.fit(train_data, batch_size = batch_size, max_iters = epochs)

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
    test_data = model.testing(test_data, batch_size = batch_size)
    #print test_data
    print "Testing time: %s" % (datetime.now() - currentTime)
    currentTime = datetime.now()
    print "Total runtime: %s" % (datetime.now() - startTime) 
    print "Finishing time: %s" % datetime.now()

    #sys.stdout = orig_stdout
    #f.close()
