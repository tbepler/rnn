import theano
import theano.tensor as T

def cross_entropy(Yh, Y):
    cent = T.zeros_like(Yh)
    i,j = T.mgrid[0:cent.shape[0], 0:cent.shape[1]]
    cent = T.set_subtensor(cent[i,j,Y], -Yh[i,j,Y])
    return cent

def confusion(Yh, Y, n):
    shape = list(Yh.shape) + [n, n]
    C = T.zeros(shape, dtype='int64')
    i,j = T.mgrid[0:C.shape[0], 0:C.shape[1]]
    C = T.set_subtensor(C[i,j,Y,Yh], 1)
    return C

def accuracy(Yh, Y, step=1):
    from theano.tensor.nnet import sigmoid
    i,j = T.mgrid[0:Y.shape[0], 0:Y.shape[1]]
    P = Yh[i,j,Y]
    Yh = T.set_subtensor(Yh[i,j,Y], float('-inf'))
    M = T.max(Yh, axis=-1)
    L = T.set_subtensor(T.zeros_like(Yh)[i,j,Y], sigmoid(step*(M-P)))
    return L


