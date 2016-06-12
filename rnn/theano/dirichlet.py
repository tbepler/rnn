from softmax import logsoftmax

def dirichlet_likelihood(X, alpha=None, axis=-1):
    if alpha is None:
        alpha = 1.0 / X.shape[axis]
    X = logsoftmax(X, axis=axis)
    L = (alpha-1.0)*X
    return L.sum(axis=axis)

