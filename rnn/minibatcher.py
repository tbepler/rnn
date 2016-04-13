
import math
import numpy as np

def as_matrix(data):
    n = len(data)
    m = max(len(x) for x in data)
    dtype = data[0].dtype
    X = np.zeros((m,n), dtype=dtype)
    mask = np.ones((m,n), dtype=np.int32)
    for i in xrange(n):
        x = data[i]
        k = len(x)
        X[:k,i] = x
        mask[k:,i] = 0
    return X, mask

class BatchIter(object):
    def __init__(self, data, size, shuffle=True):
        #self.data = data
        self.X, self.mask = as_matrix(data)
        self.size = size
        self.shuffle = shuffle

    def __len__(self):
        return int(math.ceil(self.X.shape[1]/float(self.size)))
        #return int(math.ceil(len(self.data)/float(self.size)))

    def __iter__(self):
        if self.shuffle:
            #somewhat hacky way to shuffle both in place the same way
            rng_state = np.random.get_state()
            np.random.shuffle(self.X.T)
            np.random.set_state(rng_state)
            np.random.shuffle(self.mask.T)
            #random.shuffle(self.data)
        size = self.size
        for i in xrange(0, self.X.shape[1], size):
            yield self.X[:,i:i+size], self.mask[:,i:i+size]