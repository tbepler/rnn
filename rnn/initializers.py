import numpy as np
import math

def xavier(X):
    m,n = X.shape
    X[:] = np.random.randn(m,n)
    X /= math.sqrt(m)

def orthogonal(X):
    m,n = X.shape
    X[:] = np.random.randn(m,n)
    u,_,v = np.linalg.svd(X, full_matrices=False)
    if u.shape == X.shape:
        X[:] = u
    else:
        X[:] = v
    #for i in xrange(n):
        #project out all previous columns
        #for j in xrange(i):
        #    X[:,i] -= np.dot(X[:,j].T, X[:,i])*X[:,j]
        #normalize
        #X[:,i] /= np.linalg.norm(X[:,i])
        


        
