import theano.tensor as th

def crossent(coding, true):
    if 'int' in true.dtype:
        dims = true.shape
        n = coding.shape[-1]
        coding = coding.reshape((th.prod(dims),n))
        true = true.flatten()
        y = th.nnet.categorical_crossentropy(coding, true)
        return y.reshape(dims)
        #return -th.log(th.choose(true, coding))
    else:
        return -th.sum(th.log(coding)*true, axis=coding.dims-1)
        
