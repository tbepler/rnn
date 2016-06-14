
import math
import numpy as np
import random

def as_matrix(data):
    n = len(data)
    m = max(len(x) for x in data)
    dtype = data[0].dtype
    X = np.zeros((m,n), dtype=dtype)
    mask = np.ones((m,n), dtype=np.float32)
    for i in xrange(n):
        x = data[i]
        k = len(x)
        X[:k,i] = x
        mask[k:,i] = 0
    return X, mask

class BatchIter(object):
    def __init__(self, data, size, shuffle=True, mask=True):
        self.data = data
        self.use_mask = mask
        #self.X, self.mask = as_matrix(data)
        self.size = size
        self.shuffle = shuffle

    @property
    def dtype(self):
        if type(self.data[0]) is tuple:
            return self.data[0][0].dtype
        return self.data[0].dtype

    def __len__(self):
        #return int(math.ceil(self.X.shape[1]/float(self.size)))
        return int(math.ceil(len(self.data)/float(self.size)))

    def __iter__(self):
        if self.shuffle:
            #somewhat hacky way to shuffle both in place the same way
            #rng_state = np.random.get_state()
            #np.random.shuffle(self.X.T)
            #np.random.set_state(rng_state)
            #np.random.shuffle(self.mask.T)
            random.shuffle(self.data)
        size = self.size
        masks = None
        if self.use_mask and type(self.data[0]) is tuple:
            data, masks = zip(*self.data)
        else:
            data = self.data
        dtype = data[0].dtype
        m = max(len(x) for x in data)
        X = np.zeros((m, size), dtype=dtype)
        if self.use_mask:
            mask = np.ones((m, size), dtype=np.int8)
        for i in xrange(0, len(data), size):
            n = min(len(data)-i, size)
            for j in xrange(n):
                x = data[i+j]
                k = len(x)
                X[:k,j] = x
                if self.use_mask:
                    if masks is not None:
                        mask[:k,j] = masks[i+j]
                    else:
                        mask[:k,j] = 1
                    mask[k:,j] = 0
            m = max(len(x) for x in data[i:i+n])
            if self.use_mask:
                yield X[:m,:n], mask[:m,:n]
            else:
                yield X[:m,:n]
            #yield self.X[:,i:i+size], self.mask[:,i:i+size]
