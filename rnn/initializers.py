import numpy as np
import math

def xavier(X):
    m,n = X.shape[0], X.shape[1]
    X /= math.sqrt(m+n-1)

def orthogonal(X):
    m,n = X.shape
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
        


        
