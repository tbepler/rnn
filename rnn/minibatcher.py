import math
import numpy as np
import random

class BatchIter(object):
    def __init__(self, data, size, shuffle=True, mask=True, xydata=True, single_out=True):
        self.xydata = xydata
        self.size = size
        self.use_mask = mask
        self.data = data
        if self.xydata:
        # if in the form ([X1, X2, X3, ...], [Y1, Y2, Y3, ...]) where each element is batched data, this should separate the data into [X,Y] batches of the form ([X1, Y1], [X2, Y2], ...)
            if self.use_mask:
                xmask, x = self.masking(data[0], single_out)
                ymask, y = self.masking(data[1], single_out)
                self.mask = zip(xmask, ymask)
            #self.data = np.concatenate((data), axis = 1).resize(data.size/(2*size) + 1,2,size)
            else:
                self.mask = None
                x = data[0]
                y = data[1]
            self.data = zip(x, y)
        else:
            if self.use_mask:
                xmask, x = self.masking(data, single_out)
                self.mask = xmask
                self.data = x
            else:
                self.mask = None
                self.data = data
        #self.X, self.mask = as_matrix(data)
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
            #np.rann dom.shuffle(self.mask.T)
            if self.use_mask:
                shuffle_data = zip(self.data,self.mask)
            else:
                shuffle_data = self.data
            np.random.shuffle(shuffle_data)
            if self.use_mask:
                self.data, self.mask = zip(*shuffle_data)
            else:
                self.data = shuffle_data
        size = self.size
        for i in xrange(0, len(self.data), size):
            x, y = zip(*self.data[i:i+size])
            mask = zip(*self.mask[i:i+size])[0]
            x = np.transpose(x).tolist()
            y = np.transpose(y).tolist()
            mask = np.transpose(mask).tolist()
            if self.use_mask:
                #print "X: %s" % x
                #print "Y: %s" % y
                #print "Mask: %s" % mask
                #print np.array(x).shape
                #print np.array(y).shape
                #print np.array(mask).shape
                yield x, y, mask 
            else:
                yield x, y
        # masks = None
        # if self.use_mask and type(self.data[0]) is tuple:
        #     data, masks = zip(*self.data)
        # else:
        #     data = self.data
        # dtype = self.dtype
        # m = max(len(x) for x in data)
        # X = np.zeros((m, size), dtype=dtype)
        # if self.use_mask:
        #     mask = np.ones((m, size), dtype=dtype)
        # for i in xrange(0, len(data), size):
        #     n = min(len(data)-i, size)
        #     for j in xrange(n):
        #         x = data[i+j]
        #         k = len(x)
        #         X[:k,j] = x
        #         if self.use_mask:
        #             if masks is not None:
        #                 mask[:k,j] = masks[i+j]
        #             else:
        #                 mask[:k,j] = 1
        #             mask[k:,j] = 0
        #     m = max(len(x) for x in data[i:i+n])
        #     if self.use_mask:
        #         yield X[:m,:n], mask[:m,:n]
        #     else:
        #         yield X[:m,:n]
        #     #yield self.X[:,i:i+size], self.mask[:,i:i+size]

    def masking(self, data, single_out):
        dtype = self.dtype
        batch_count = int(math.ceil(len(data)/float(self.size)))
        rows_needed = batch_count*self.size - len(data)
        #print data.shape
        #print [len(a) for a in data]
        #print "Size array: %s" % unpadded_size_array
        print data.shape
        if single_out:
            size_array = np.append(np.array([len(data[i]) for i in range(len(data))]), np.zeros(rows_needed))
            mask = np.arange(max(size_array)) < size_array[:,None]
            zero_mask = np.zeros(mask.shape, dtype = dtype)
            zero_mask[mask] = np.hstack((data[:]))
        else:
            unpadded_size_array = np.reshape(np.array([len(i[1]) for i in list(np.ndenumerate(data))]), data.shape)
            if len(data.shape) == 1:
                size_array = np.pad(unpadded_size_array, ((0,rows_needed)), mode="constant", constant_values=0)
                mask = np.arange(np.array(max(array))) < size_array[:,None]
                print "Mask shape:"
                print mask.shape
                zero_mask = np.zeros(mask.shape, dtype = dtype)
                #print "Zero mask: %s" % zero_mask
                #print np.hstack((data[:,:]))
                #print mask
                zero_mask[mask] = np.hstack(np.hstack((data[:])))
                #print "Zero shape: "
                #print zero_mask.shape
            else:
                unpadded_size_array = np.reshape(np.array([len(i[1]) for i in list(np.ndenumerate(data))]), data.shape)
                size_array = np.pad(unpadded_size_array, ((0,rows_needed),(0,0)), mode="constant", constant_values=0)
                mask = np.arange(np.array(max([max(array) for array in size_array]))) < size_array[:,:,None]
                #print mask.shape
                #print "Mask shape:"
                #print mask.shape
                zero_mask = np.zeros(mask.shape, dtype = dtype)
                #print "Zero mask: %s" % zero_mask
                #print np.hstack((data[:,:]))
                #print mask
                zero_mask[mask] = np.hstack(np.hstack((data[:,:])))
                #print "Zero shape: "
                #print zero_mask.shape
        #print zero_mask
        #print data
        #mask = np.invert(mask)
        mask = mask.astype(np.int8)
        return 1*mask, zero_mask
