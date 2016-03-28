import ctypes
import numpy as np
import numpy.ctypeslib as npct
import math

lib = ctypes.cdll.LoadLibrary('lib/librnn.so')
lib.sicentfw.restype = ctypes.c_float
lib.slcentfw.restype = ctypes.c_float
lib.dicentfw.restype = ctypes.c_double
lib.dlcentfw.restype = ctypes.c_double

def as_ptr(x, eltype):
    return x.ctypes.data_as(ctypes.POINTER(eltype))

def leading_dim(X):
    return X.strides[-2]/X.itemsize

def lstmfw(W, X, S, Y):
    #check dimensions
    (p,batches,m) = Y.shape
    p -= 1
    n = X.shape[2]
    assert X.shape == (p,batches,n)
    assert W.shape == (m+n+1, 4*m)
    assert S.shape == (p+1,batches,6*m)

    ldx = X.strides[1]/X.itemsize
    lds = S.strides[1]/S.itemsize
    ldy = Y.strides[1]/Y.itemsize
    
    #dispatch function by type
    assert W.dtype == X.dtype == S.dtype == Y.dtype
    if W.dtype == np.float32:
        float_t = ctypes.c_float
        f = lib.slstmfw
    elif W.dtype == np.float64:
        float_t = ctypes.c_double
        f = lib.dlstmfw
    else:
        raise Exception("lstmfw: unsupported type, {}".format(W.dtype))

    f(ctypes.c_int(m)
      , ctypes.c_int(n)
      , ctypes.c_int(batches)
      , ctypes.c_int(p)
      , X.ctypes.data_as(ctypes.POINTER(float_t))
      , ctypes.c_int(ldx)
      , W.ctypes.data_as(ctypes.POINTER(float_t))
      , Y.ctypes.data_as(ctypes.POINTER(float_t))
      , ctypes.c_int(ldy)
      , S.ctypes.data_as(ctypes.POINTER(float_t))
      , ctypes.c_int(lds))

def lstmbw(tau, W, X, S, Y, dY, dW, dX, dS):
    #check dimensions
    (p,batches,n) = X.shape
    m = Y.shape[2]
    assert Y.shape == (p+1,batches,m)
    assert W.shape == (m+n+1, 4*m)
    assert S.shape == (p+1,batches,6*m)
    assert W.shape == dW.shape
    assert X.shape == dX.shape
    assert dS.shape == S.shape

    #dispatch function by type
    assert W.dtype == X.dtype == S.dtype == Y.dtype == dY.dtype == dW.dtype == dX.dtype == dS.dtype
    if W.dtype == np.float32:
        float_t = ctypes.c_float
        f = lib.slstmbw
    elif W.dtype == np.float64:
        float_t = ctypes.c_double
        f = lib.dlstmbw
    else:
        raise Exception("lstmbw: unsupported type, {}".format(W.dtype))
    
    ldx = ctypes.c_int(X.strides[1]/X.itemsize)
    lds = ctypes.c_int(S.strides[1]/S.itemsize)
    ldy = ctypes.c_int(Y.strides[1]/Y.itemsize)
    lddy = ctypes.c_int(dY.strides[1]/dY.itemsize)
    lddx = ctypes.c_int(dX.strides[1]/dX.itemsize)
    ldds = ctypes.c_int(dS.strides[1]/dS.itemsize)

    f(float_t(tau)
      , ctypes.c_int(m)
      , ctypes.c_int(n)
      , ctypes.c_int(batches)
      , ctypes.c_int(p)
      , as_ptr(X, float_t)
      , ldx
      , as_ptr(W, float_t)
      , as_ptr(Y, float_t)
      , ldy
      , as_ptr(S, float_t)
      , lds
      , as_ptr(dY, float_t)
      , lddy
      , as_ptr(dX, float_t)
      , lddx
      , as_ptr(dW, float_t)
      , as_ptr(dS, float_t)
      , ldds)

def lstmencbw(tau, W, X, S, Y, dY, dW, dX, dS):
    #check dimensions
    (p,batches,n) = X.shape
    m = Y.shape[2]
    assert Y.shape == (p+1,batches,m)
    assert W.shape == (m+n+1, 4*m)
    assert S.shape == (p+1,batches,6*m)
    assert W.shape == dW.shape
    assert X.shape == dX.shape
    assert dS.shape == S.shape
    assert dY.shape == (1,batches,m)

    #dispatch function by type
    assert W.dtype == X.dtype == S.dtype == Y.dtype == dY.dtype == dW.dtype == dX.dtype == dS.dtype
    if W.dtype == np.float32:
        float_t = ctypes.c_float
        f = lib.slstmencbw
    elif W.dtype == np.float64:
        float_t = ctypes.c_double
        f = lib.dlstmencbw
    else:
        raise Exception("lstmencbw: unsupported type, {}".format(W.dtype))
    
    ldx = ctypes.c_int(X.strides[1]/X.itemsize)
    lds = ctypes.c_int(S.strides[1]/S.itemsize)
    ldy = ctypes.c_int(Y.strides[1]/Y.itemsize)
    lddy = ctypes.c_int(dY.strides[1]/dY.itemsize)
    lddx = ctypes.c_int(dX.strides[1]/dX.itemsize)
    ldds = ctypes.c_int(dS.strides[1]/dS.itemsize)

    f(float_t(tau)
      , ctypes.c_int(m)
      , ctypes.c_int(n)
      , ctypes.c_int(batches)
      , ctypes.c_int(p)
      , as_ptr(X, float_t)
      , ldx
      , as_ptr(W, float_t)
      , as_ptr(Y, float_t)
      , ldy
      , as_ptr(S, float_t)
      , lds
      , as_ptr(dY, float_t)
      , lddy
      , as_ptr(dX, float_t)
      , lddx
      , as_ptr(dW, float_t)
      , as_ptr(dS, float_t)
      , ldds)

def lstmrfw(W, X, S, Y):
    #check dimensions
    (p,batches,m) = Y.shape
    p -= 1
    n = X.shape[2]
    assert X.shape == (p,batches,n)
    assert W.shape == (m+n+1, 4*m)
    assert S.shape == (p+1,batches,6*m)

    ldx = X.strides[1]/X.itemsize
    lds = S.strides[1]/S.itemsize
    ldy = Y.strides[1]/Y.itemsize
    
    #dispatch function by type
    assert W.dtype == X.dtype == S.dtype == Y.dtype
    if W.dtype == np.float32:
        float_t = ctypes.c_float
        f = lib.slstmrfw
    elif W.dtype == np.float64:
        float_t = ctypes.c_double
        f = lib.dlstmrfw
    else:
        raise Exception("lstmrfw: unsupported type, {}".format(W.dtype))

    f(ctypes.c_int(m)
      , ctypes.c_int(n)
      , ctypes.c_int(batches)
      , ctypes.c_int(p)
      , X.ctypes.data_as(ctypes.POINTER(float_t))
      , ctypes.c_int(ldx)
      , W.ctypes.data_as(ctypes.POINTER(float_t))
      , Y.ctypes.data_as(ctypes.POINTER(float_t))
      , ctypes.c_int(ldy)
      , S.ctypes.data_as(ctypes.POINTER(float_t))
      , ctypes.c_int(lds))

def lstmrbw(tau, W, X, S, Y, dY, dW, dX, dS):
    #check dimensions
    (p,batches,n) = X.shape
    m = Y.shape[2]
    assert Y.shape == (p+1,batches,m)
    assert W.shape == (m+n+1, 4*m)
    assert S.shape == (p+1,batches,6*m)
    assert W.shape == dW.shape
    assert X.shape == dX.shape
    assert dS.shape == S.shape

    #dispatch function by type
    assert W.dtype == X.dtype == S.dtype == Y.dtype == dY.dtype == dW.dtype == dX.dtype == dS.dtype
    if W.dtype == np.float32:
        float_t = ctypes.c_float
        f = lib.slstmrbw
    elif W.dtype == np.float64:
        float_t = ctypes.c_double
        f = lib.dlstmrbw
    else:
        raise Exception("lstmrbw: unsupported type, {}".format(W.dtype))
    
    ldx = ctypes.c_int(X.strides[1]/X.itemsize)
    lds = ctypes.c_int(S.strides[1]/S.itemsize)
    ldy = ctypes.c_int(Y.strides[1]/Y.itemsize)
    lddy = ctypes.c_int(dY.strides[1]/dY.itemsize)
    lddx = ctypes.c_int(dX.strides[1]/dX.itemsize)
    ldds = ctypes.c_int(dS.strides[1]/dS.itemsize)

    f(float_t(tau)
      , ctypes.c_int(m)
      , ctypes.c_int(n)
      , ctypes.c_int(batches)
      , ctypes.c_int(p)
      , as_ptr(X, float_t)
      , ldx
      , as_ptr(W, float_t)
      , as_ptr(Y, float_t)
      , ldy
      , as_ptr(S, float_t)
      , lds
      , as_ptr(dY, float_t)
      , lddy
      , as_ptr(dX, float_t)
      , lddx
      , as_ptr(dW, float_t)
      , as_ptr(dS, float_t)
      , ldds)

"""
def bilstmfw(W, X, S, Y):
    #check dimensions
    (p,batches,m) = Y.shape
    m /= 2
    p -= 2
    n = X.shape[2]
    assert X.shape == (p,batches,n)
    assert W.shape == (2*(m+n+1), 4*m)
    assert S.shape == (p+2,batches,12*m)

    ldx = X.strides[1]/X.itemsize
    lds = S.strides[1]/S.itemsize
    ldy = Y.strides[1]/Y.itemsize
    
    #dispatch function by type
    assert W.dtype == X.dtype == S.dtype == Y.dtype
    if W.dtype == np.float32:
        float_t = ctypes.c_float
        f = lib.sbilstmfw
    elif W.dtype == np.float64:
        float_t = ctypes.c_double
        f = lib.dbilstmfw
    else:
        raise Exception("bilstmfw: unsupported type, {}".format(W.dtype))

    f(ctypes.c_int(m)
      , ctypes.c_int(n)
      , ctypes.c_int(batches)
      , ctypes.c_int(p)
      , X.ctypes.data_as(ctypes.POINTER(float_t))
      , ctypes.c_int(ldx)
      , W.ctypes.data_as(ctypes.POINTER(float_t))
      , Y.ctypes.data_as(ctypes.POINTER(float_t))
      , ctypes.c_int(ldy)
      , S.ctypes.data_as(ctypes.POINTER(float_t))
      , ctypes.c_int(lds))

    print "Returning from bilstmfw"
    import sys
    sys.stdout.flush()

def bilstmbw(tau, W, X, S, Y, dY, dW, dX, dS):
    #check dimensions
    (p,batches,n) = X.shape
    m = Y.shape[2]/2
    assert Y.shape == (p+2,batches,2*m)
    assert W.shape == (2*(m+n+1), 4*m)
    assert S.shape == (p+2,batches,12*m)
    assert W.shape == dW.shape
    assert X.shape == dX.shape
    assert dS.shape == S.shape

    #dispatch function by type
    assert W.dtype == X.dtype == S.dtype == Y.dtype == dY.dtype == dW.dtype == dX.dtype == dS.dtype
    if W.dtype == np.float32:
        float_t = ctypes.c_float
        f = lib.sbilstmbw
    elif W.dtype == np.float64:
        float_t = ctypes.c_double
        f = lib.dbilstmbw
    else:
        raise Exception("bilstmbw: unsupported type, {}".format(W.dtype))
    
    ldx = ctypes.c_int(X.strides[1]/X.itemsize)
    lds = ctypes.c_int(S.strides[1]/S.itemsize)
    ldy = ctypes.c_int(Y.strides[1]/Y.itemsize)
    lddy = ctypes.c_int(dY.strides[1]/dY.itemsize)
    lddx = ctypes.c_int(dX.strides[1]/dX.itemsize)
    ldds = ctypes.c_int(dS.strides[1]/dS.itemsize)

    f(float_t(tau)
      , ctypes.c_int(m)
      , ctypes.c_int(n)
      , ctypes.c_int(batches)
      , ctypes.c_int(p)
      , as_ptr(X, float_t)
      , ldx
      , as_ptr(W, float_t)
      , as_ptr(Y, float_t)
      , ldy
      , as_ptr(S, float_t)
      , lds
      , as_ptr(dY, float_t)
      , lddy
      , as_ptr(dX, float_t)
      , lddx
      , as_ptr(dW, float_t)
      , as_ptr(dS, float_t)
      , ldds)
"""

def emlstmfw(W, X, S, Y):
    #check dimensions
    (p,batches,m) = Y.shape
    p -= 1
    n = W.shape[0]-m-1
    assert X.shape == (p,batches)
    assert W.shape == (m+n+1, 4*m)
    assert S.shape == (p+1,batches,6*m)

    ldx = leading_dim(X)
    lds = S.strides[1]/S.itemsize
    ldy = Y.strides[1]/Y.itemsize

    #type dispatch
    assert W.dtype == Y.dtype == S.dtype
    if W.dtype == np.float32:
        float_t = ctypes.c_float
        if X.dtype == np.int32:
            int_t = ctypes.c_int
            f = lib.siemlstmfw
        elif X.dtype == np.int64:
            int_t = ctypes.c_long
            f = lib.slemlstmfw
        else:
            raise Exception("emlstmfw: unsupported int type, {}".format(X.dtype))
    elif W.dtype == np.float64:
        float_t = ctypes.c_double
        if X.dtype == np.int32:
            int_t = ctypes.c_int
            f = lib.diemlstmfw
        elif X.dtype == np.int64:
            int_t = ctypes.c_long
            f = lib.dlemlstmfw
        else:
            raise Exception("emlstmfw: unsupported int type, {}".format(X.dtype))
    else:
        raise Exception("emlstmfw: unsupported float type, {}".format(W.dtype))
    
    f(ctypes.c_int(m)
      , ctypes.c_int(n)
      , ctypes.c_int(batches)
      , ctypes.c_int(p)
      , X.ctypes.data_as(ctypes.POINTER(int_t))
      , ctypes.c_int(ldx)
      , W.ctypes.data_as(ctypes.POINTER(float_t))
      , Y.ctypes.data_as(ctypes.POINTER(float_t))
      , ctypes.c_int(ldy)
      , S.ctypes.data_as(ctypes.POINTER(float_t))
      , ctypes.c_int(lds))

def emlstmbw(tau, W, X, S, Y, dY, dW, dS):
    #check dimensions
    (p,batches,m) = Y.shape
    p -= 1
    n = W.shape[0]-m-1
    assert Y.shape == (p+1,batches,m)
    assert W.shape == (m+n+1, 4*m)
    assert S.shape == (p+1,batches,6*m)
    assert W.shape == dW.shape
    assert X.shape == (p,batches)
    assert dS.shape == S.shape

    #type dispatch
    assert W.dtype == Y.dtype == S.dtype == dY.dtype == dW.dtype == dS.dtype
    if W.dtype == np.float32:
        float_t = ctypes.c_float
        if X.dtype == np.int32:
            int_t = ctypes.c_int
            f = lib.siemlstmbw
        elif X.dtype == np.int64:
            int_t = ctypes.c_long
            f = lib.slemlstmbw
        else:
            raise Exception("emlstmbw: unsupported int type, {}".format(X.dtype))
    elif W.dtype == np.float64:
        float_t = ctypes.c_double
        if X.dtype == np.int32:
            int_t = ctypes.c_int
            f = lib.diemlstmbw
        elif X.dtype == np.int64:
            int_t = ctypes.c_long
            f = lib.dlemlstmbw
        else:
            raise Exception("emlstmbw: unsupported int type, {}".format(X.dtype))
    else:
        raise Exception("emlstmbw: unsupported float type, {}".format(W.dtype))

    ldx = ctypes.c_int(leading_dim(X))
    lds = ctypes.c_int(S.strides[1]/S.itemsize)
    ldy = ctypes.c_int(Y.strides[1]/Y.itemsize)
    lddy = ctypes.c_int(dY.strides[1]/dY.itemsize)
    ldds = ctypes.c_int(dS.strides[1]/dS.itemsize)

    f(float_t(tau)
      , ctypes.c_int(m)
      , ctypes.c_int(n)
      , ctypes.c_int(batches)
      , ctypes.c_int(p)
      , as_ptr(X, float_t)
      , ldx
      , as_ptr(W, float_t)
      , as_ptr(Y, float_t)
      , ldy
      , as_ptr(S, float_t)
      , lds
      , as_ptr(dY, float_t)
      , lddy
      , as_ptr(dW, float_t)
      , as_ptr(dS, float_t)
      , ldds)

def emlstmencbw(tau, W, X, S, Y, dY, dW, dS):
    #check dimensions
    (p,batches,m) = Y.shape
    p -= 1
    n = W.shape[0]-m-1
    assert Y.shape == (p+1,batches,m)
    assert W.shape == (m+n+1, 4*m)
    assert S.shape == (p+1,batches,6*m)
    assert W.shape == dW.shape
    assert X.shape == (p,batches)
    assert dS.shape == S.shape
    assert dY.shape == (1,batches,m)

    #type dispatch
    assert W.dtype == Y.dtype == S.dtype == dY.dtype == dW.dtype == dS.dtype
    if W.dtype == np.float32:
        float_t = ctypes.c_float
        if X.dtype == np.int32:
            int_t = ctypes.c_int
            f = lib.siemlstmencbw
        elif X.dtype == np.int64:
            int_t = ctypes.c_long
            f = lib.slemlstmencbw
        else:
            raise Exception("emlstmencbw: unsupported int type, {}".format(X.dtype))
    elif W.dtype == np.float64:
        float_t = ctypes.c_double
        if X.dtype == np.int32:
            int_t = ctypes.c_int
            f = lib.diemlstmencbw
        elif X.dtype == np.int64:
            int_t = ctypes.c_long
            f = lib.dlemlstmencbw
        else:
            raise Exception("emlstmencbw: unsupported int type, {}".format(X.dtype))
    else:
        raise Exception("emlstmencbw: unsupported float type, {}".format(W.dtype))

    ldx = ctypes.c_int(leading_dim(X))
    lds = ctypes.c_int(S.strides[1]/S.itemsize)
    ldy = ctypes.c_int(Y.strides[1]/Y.itemsize)
    lddy = ctypes.c_int(dY.strides[1]/dY.itemsize)
    ldds = ctypes.c_int(dS.strides[1]/dS.itemsize)

    f(float_t(tau)
      , ctypes.c_int(m)
      , ctypes.c_int(n)
      , ctypes.c_int(batches)
      , ctypes.c_int(p)
      , as_ptr(X, float_t)
      , ldx
      , as_ptr(W, float_t)
      , as_ptr(Y, float_t)
      , ldy
      , as_ptr(S, float_t)
      , lds
      , as_ptr(dY, float_t)
      , lddy
      , as_ptr(dW, float_t)
      , as_ptr(dS, float_t)
      , ldds)

def linearfw(W, X, Y):
    #dimension check
    (k,batches,n) = X.shape
    m = Y.shape[2]
    assert Y.shape == (k,batches,m)
    assert W.shape == (n+1,m)

    #type dispatch
    assert W.dtype == X.dtype == Y.dtype
    if W.dtype == np.float32:
        float_t = ctypes.c_float
        f = lib.slinearfw
    elif W.dtype == np.float64:
        float_t = ctypes.c_double
        f = lib.dlinearfw
    else:
        raise Exception("linearfw: unsupported type, {}".format(W.dtype))

    m = ctypes.c_int(m)
    n = ctypes.c_int(n)
    p = ctypes.c_int(k*batches)
    ldx = ctypes.c_int(leading_dim(X))
    ldy = ctypes.c_int(leading_dim(Y))

    f(m, n, p, as_ptr(W,float_t), as_ptr(X,float_t), ldx, as_ptr(Y,float_t), ldy)

def linearbw(W, X, dY, dW, dX):
    #dimension check
    (k,batches,n) = X.shape
    m = dY.shape[2]
    assert dY.shape == (k,batches,m)
    assert W.shape == (n+1,m)
    assert dW.shape == W.shape
    assert dX.shape == X.shape

    #type dispatch
    assert W.dtype == X.dtype == dY.dtype == dW.dtype == dX.dtype
    if W.dtype == np.float32:
        float_t = ctypes.c_float
        f = lib.slinearbw
    elif W.dtype == np.float64:
        float_t = ctypes.c_double
        f = lib.dlinearbw
    else:
        raise Exception("linearbw: unsupported type, {}".format(W.dtype))

    m = ctypes.c_int(m)
    n = ctypes.c_int(n)
    p = ctypes.c_int(k*batches)
    ldx = ctypes.c_int(leading_dim(X))
    lddy = ctypes.c_int(leading_dim(dY))
    lddx = ctypes.c_int(leading_dim(dX))

    f(m, n, p, as_ptr(W,float_t), as_ptr(X,float_t), ldx, as_ptr(dY,float_t), lddy
      , as_ptr(dW,float_t), as_ptr(dX,float_t), lddx)

def embedfw(W, X, Y):
    #dimension check
    (k,batches) = X.shape
    (n,m) = W.shape
    assert Y.shape == (k,batches,m)

    #type dispatch
    assert W.dtype == Y.dtype
    if W.dtype == np.float32:
        float_t = ctypes.c_float
        if X.dtype == np.int32:
            int_t = ctypes.c_int
            f = lib.isembedfw
        elif X.dtype == np.int64:
            int_t = ctypes.c_long
            f = lib.lsembedfw
        else:
            raise Exception("embedfw: unsupported int type, {}".format(X.dtype))
    elif W.dtype == np.float64:
        float_t = ctypes.c_double
        if X.dtype == np.int32:
            int_t = ctypes.c_int
            f = lib.idembedfw
        elif X.dtype == np.int64:
            int_t = ctypes.c_long
            f = lib.ldembedfw
        else:
            raise Exception("embedfw: unsupported int type, {}".format(X.dtype))
    else:
        raise Exception("embedfw: unsupported float type, {}".format(W.dtype))

    m = ctypes.c_int(m)
    n = ctypes.c_int(n)
    b = ctypes.c_int(batches)
    k = ctypes.c_int(k)
    ldx = ctypes.c_int(leading_dim(X))
    ldy = ctypes.c_int(leading_dim(Y))

    f(m, n, b, k, as_ptr(W,float_t), as_ptr(X,int_t), ldx, as_ptr(Y,float_t), ldy)

def embedbw(X, dY, dW):
    #dimension check
    (k,batches) = X.shape
    (n,m) = dW.shape
    assert dY.shape == (k,batches,m)

    #type dispatch
    assert dW.dtype == dY.dtype
    if dW.dtype == np.float32:
        float_t = ctypes.c_float
        if X.dtype == np.int32:
            int_t = ctypes.c_int
            f = lib.isembedbw
        elif X.dtype == np.int64:
            int_t = ctypes.c_long
            f = lib.lsembedbw
        else:
            raise Exception("embedbw: unsupported int type, {}".format(X.dtype))
    elif dW.dtype == np.float64:
        float_t = ctypes.c_double
        if X.dtype == np.int32:
            int_t = ctypes.c_int
            f = lib.idembedbw
        elif X.dtype == np.int64:
            int_t = ctypes.c_long
            f = lib.ldembedbw
        else:
            raise Exception("embedbw: unsupported int type, {}".format(X.dtype))
    else:
        raise Exception("embedbw: unsupported float type, {}".format(dW.dtype))

    m = ctypes.c_int(m)
    n = ctypes.c_int(n)
    b = ctypes.c_int(batches)
    k = ctypes.c_int(k)
    ldx = ctypes.c_int(leading_dim(X))
    lddy = ctypes.c_int(leading_dim(dY))

    f(m, n, b, k, as_ptr(X,int_t), ldx, as_ptr(dY,float_t), lddy, as_ptr(dW,float_t))

def convfw(W, X, Y):
    #dimension check
    (k,batches,n) = X.shape
    m = Y.shape[2]
    p = W.shape[0]/n
    assert Y.shape == (k+p-1,batches,m)
    assert W.shape == (n*p,m)

    #type dispatch
    assert W.dtype == X.dtype == Y.dtype
    if W.dtype == np.float32:
        float_t = ctypes.c_float
        f = lib.sconvfw
    elif W.dtype == np.float64:
        float_t = ctypes.c_double
        f = lib.dconvfw
    else:
        raise Exception("convfw: unsupported type, {}".format(W.dtype))

    m = ctypes.c_int(m)
    p = ctypes.c_int(p)
    n = ctypes.c_int(n)
    k = ctypes.c_int(k)
    b = ctypes.c_int(batches)
    ldx = ctypes.c_int(leading_dim(X))
    ldy = ctypes.c_int(leading_dim(Y))

    f(m, p, n, k, b, as_ptr(W,float_t), as_ptr(X,float_t), ldx, as_ptr(Y,float_t), ldy)

def convbw(W, X, dY, dW, dX):
    #dimension check
    (k,batches,n) = X.shape
    m = dY.shape[2]
    p = W.shape[0]/n
    assert dY.shape == (k,batches,m)
    assert W.shape == (n*p,m)
    assert dW.shape == W.shape
    assert dX.shape == X.shape

    #type dispatch
    assert W.dtype == X.dtype == dY.dtype == dW.dtype == dX.dtype
    if W.dtype == np.float32:
        float_t = ctypes.c_float
        f = lib.sconvbw
    elif W.dtype == np.float64:
        float_t = ctypes.c_double
        f = lib.dconvbw
    else:
        raise Exception("convbw: unsupported type, {}".format(W.dtype))

    m = ctypes.c_int(m)
    p = ctypes.c_int(p)
    n = ctypes.c_int(n)
    k = ctypes.c_int(k)
    b = ctypes.c_int(batches)
    ldx = ctypes.c_int(leading_dim(X))
    lddy = ctypes.c_int(leading_dim(dY))
    lddx = ctypes.c_int(leading_dim(dX))

    f(m, p, n, k, b, as_ptr(W,float_t), as_ptr(X,float_t), ldx, as_ptr(dY,float_t), lddy
      , as_ptr(dW,float_t), as_ptr(dX,float_t), lddx)


def emconvfw(W, X, Y):
    #dimension check
    (k,batches) = X.shape
    m = Y.shape[2]
    p = Y.shape[0]-k+1
    n = W.shape[0]/p
    assert W.shape == (n*p,m)
    assert Y.shape == (k+p-1,batches,m)

    #type dispatch
    assert W.dtype == Y.dtype
    if W.dtype == np.float32:
        float_t = ctypes.c_float
        if X.dtype == np.int32:
            int_t = ctypes.c_int
            f = lib.isemconvfw
        elif X.dtype == np.int64:
            int_t = ctypes.c_long
            f = lib.lsemconvfw
        else:
            raise Exception("emconvfw: unsupported int type, {}".format(X.dtype))
    elif W.dtype == np.float64:
        float_t = ctypes.c_double
        if X.dtype == np.int32:
            int_t = ctypes.c_int
            f = lib.idemconvfw
        elif X.dtype == np.int64:
            int_t = ctypes.c_long
            f = lib.ldemconvfw
        else:
            raise Exception("emconvfw: unsupported int type, {}".format(X.dtype))
    else:
        raise Exception("emconvfw: unsupported float type, {}".format(W.dtype))

    m = ctypes.c_int(m)
    p = ctypes.c_int(p)
    n = ctypes.c_int(n)
    b = ctypes.c_int(batches)
    k = ctypes.c_int(k)
    ldx = ctypes.c_int(leading_dim(X))
    ldy = ctypes.c_int(leading_dim(Y))

    f(m, p, n, k, b, as_ptr(W,float_t), as_ptr(X,int_t), ldx, as_ptr(Y,float_t), ldy)

def emconvbw(n, X, dY, dW):
    #dimension check
    (k,batches) = X.shape
    m = dW.shape[1]
    p = dW.shape[0]/n
    assert dY.shape == (k,batches,m)

    #type dispatch
    assert dW.dtype == dY.dtype
    if dW.dtype == np.float32:
        float_t = ctypes.c_float
        if X.dtype == np.int32:
            int_t = ctypes.c_int
            f = lib.isemconvbw
        elif X.dtype == np.int64:
            int_t = ctypes.c_long
            f = lib.lsemconvbw
        else:
            raise Exception("emconvbw: unsupported int type, {}".format(X.dtype))
    elif dW.dtype == np.float64:
        float_t = ctypes.c_double
        if X.dtype == np.int32:
            int_t = ctypes.c_int
            f = lib.idemconvbw
        elif X.dtype == np.int64:
            int_t = ctypes.c_long
            f = lib.ldemconvbw
        else:
            raise Exception("emconvbw: unsupported int type, {}".format(X.dtype))
    else:
        raise Exception("emconvbw: unsupported float type, {}".format(dW.dtype))

    m = ctypes.c_int(m)
    p = ctypes.c_int(p)
    n = ctypes.c_int(n)
    b = ctypes.c_int(batches)
    k = ctypes.c_int(k)
    ldx = ctypes.c_int(leading_dim(X))
    lddy = ctypes.c_int(leading_dim(dY))

    f(m, p, n, k, b, as_ptr(X,int_t), ldx, as_ptr(dY,float_t), lddy, as_ptr(dW,float_t))

    
def softmaxfw(X, Y):
    (k,batches,m) = X.shape
    assert X.shape == Y.shape

    #type dispatch
    assert X.dtype == Y.dtype
    if X.dtype == np.float32:
        float_t = ctypes.c_float
        f = lib.ssoftmaxfw
    elif X.dtype == np.float64:
        float_t = ctypes.c_double
        f = lib.dsoftmaxfw
    else:
        raise Exception("softmaxfw: unsupported type, {}".format(X.dtype))

    m = ctypes.c_int(m)
    n = ctypes.c_int(k*batches)
    ldx = ctypes.c_int(leading_dim(X))
    ldy = ctypes.c_int(leading_dim(Y))

    f(m, n, as_ptr(X,float_t), ldx, as_ptr(Y,float_t), ldy)

def centfw(Yh, Y):
    (k,batches,m) = Yh.shape
    assert Y.shape == (k,batches)

    #type dispatch
    if Yh.dtype == np.float32:
        float_t = ctypes.c_float
        if Y.dtype == np.int32:
            int_t = ctypes.c_int
            f = lib.sicentfw
        elif Y.dtype == np.int64:
            int_t = ctypes.c_long
            f = lib.slcentfw
        else:
            raise Exception("centfw: unsupported int type, {}".format(Y.dtype))
    elif Yh.dtype == np.float64:
        float_t = ctypes.c_double
        if Y.dtype == np.int32:
            int_t = ctypes.c_int
            f = lib.dicentfw
        elif Y.dtype == np.int64:
            int_t = ctypes.c_long
            f = lib.dlcentfw
        else:
            raise Exception("centfw: unsupported int type, {}".format(Y.dtype))
    else:
        raise Exception("centfw: unsupported float type, {}".format(Yh.dtype))

    m = ctypes.c_int(m)
    b = ctypes.c_int(batches)
    k = ctypes.c_int(k)
    ldyh = ctypes.c_int(leading_dim(Yh))
    ldy = ctypes.c_int(leading_dim(Y))

    return f(m, b, k, as_ptr(Yh,float_t), ldyh, as_ptr(Y,int_t), ldy)

def entmaxbw(Yh, Y, dX):
    (k,batches,m) = Yh.shape
    assert Yh.shape == dX.shape
    assert Y.shape == (k,batches)

    #type dispatch
    assert Yh.dtype == dX.dtype
    if Yh.dtype == np.float32:
        float_t = ctypes.c_float
        if Y.dtype == np.int32:
            int_t = ctypes.c_int
            f = lib.sientmaxbw
        elif Y.dtype == np.int64:
            int_t = ctypes.c_long
            f = lib.slentmaxbw
        else:
            raise Exception("centfw: unsupported int type, {}".format(Y.dtype))
    elif Yh.dtype == np.float64:
        float_t = ctypes.c_double
        if Y.dtype == np.int32:
            int_t = ctypes.c_int
            f = lib.dientmaxbw
        elif Y.dtype == np.int64:
            int_t = ctypes.c_long
            f = lib.dlentmaxbw
        else:
            raise Exception("centfw: unsupported int type, {}".format(Y.dtype))
    else:
        raise Exception("centfw: unsupported float type, {}".format(Yh.dtype))

    m = ctypes.c_int(m)
    b = ctypes.c_int(batches)
    k = ctypes.c_int(k)
    ldyh = ctypes.c_int(leading_dim(Yh))
    ldy = ctypes.c_int(leading_dim(Y))
    lddx = ctypes.c_int(leading_dim(dX))

    f(m, b, k, as_ptr(Yh,float_t), ldyh, as_ptr(Y,int_t), ldy, as_ptr(dX,float_t), lddx)

def adadelta(rho, eps, G, X, G2, dX2):
    n = X.size
    assert X.size == G.size == G2.size == dX2.size

    #type dispatch
    assert X.dtype == G.dtype == G2.dtype == dX2.dtype
    if X.dtype == np.float32:
        float_t = ctypes.c_float
        f = lib.sadadelta
    elif X.dtype == np.float64:
        float_t = ctypes.c_double
        f = lib.dadadelta
    else:
        raise Exception("adadelta: unsupported type, {}".format(X.dtype))

    n = ctypes.c_int(n)
    rho = float_t(rho)
    eps = float_t(eps)

    f(rho, eps, n, as_ptr(G,float_t), as_ptr(X,float_t), as_ptr(G2,float_t), as_ptr(dX2,float_t))

def rmsprop(nu, rho, eps, G, X, G2):
    n = X.size
    assert X.size == G.size == G2.size

    #type dispatch
    assert X.dtype == G.dtype == G2.dtype
    if X.dtype == np.float32:
        float_t = ctypes.c_float
        f = lib.srmsprop
    elif X.dtype == np.float64:
        float_t = ctypes.c_double
        f = lib.drmsprop
    else:
        raise Exception("rmsprop: unsupported type, {}".format(X.dtype))

    n = ctypes.c_int(n)
    nu = float_t(nu)
    rho = float_t(rho)
    eps = float_t(eps)

    f(nu, rho, eps, n, as_ptr(G,float_t), as_ptr(X,float_t), as_ptr(G2,float_t))

def olbfgs(nu, lmbda, eps, t, grad, x, g, s, y):
    (n,m) = s.shape
    n -= 1
    assert s.shape == y.shape
    assert x.shape == g.shape
    assert x.shape == (m,)

    #type dispatch
    assert x.dtype == g.dtype == s.dtype == y.dtype
    if x.dtype == np.float32:
        float_t = ctypes.c_float
        f = lib.solbfgs
    elif x.dtype == np.float64:
        float_t = ctypes.c_double
        f = lib.dolbfgs
    else:
        raise Exception("olbfgs: unsupported type, {}".format(X.dtype))

    yh = [None]
    callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(float_t), ctypes.POINTER(float_t))
    @callback_type
    def callback(x, g):
        x = npct.as_array(x, (m,))
        #print x[0]
        g = npct.as_array(g, (m,))
        #print g[0]
        yh[0] = grad(x, g)

    nu = float_t(nu)
    lmbda = float_t(lmbda)
    eps = float_t(eps)
    t = ctypes.c_int(t)
    c_m = ctypes.c_int(m)
    n = ctypes.c_int(n)

    f(nu, lmbda, eps, t, callback, c_m, n, as_ptr(x,float_t), as_ptr(g,float_t), as_ptr(s,float_t)
      , as_ptr(y,float_t))
    #print yh[0].shape
    #print yh[0][0:1]
    return yh[0]
        
    
def dropout(p, X, mask, flag=1):
    (k,b,m) = X.shape
    assert X.shape == mask.shape
    assert mask.dtype == np.int32
    
    p = ctypes.c_float(p)
    m = ctypes.c_int(m)
    n = ctypes.c_int(k*b)
    ldx = ctypes.c_int(leading_dim(X))
    flag = ctypes.c_int(flag)

    if X.dtype == np.float32:
        float_t = ctypes.c_float
        f = lib.sdropout
    elif X.dtype == np.float64:
        float_t = ctypes.c_double
        f = lib.ddropout
    else:
        raise Exception("dropout: unsupported type, {}".format(X.dtype))

    f(p, m, n, as_ptr(X,float_t), ldx, as_ptr(mask,ctypes.c_int), flag)

    
