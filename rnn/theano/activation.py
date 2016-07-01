
def fast_tanh(x):
    return x/(1+abs(x))

def fast_sigmoid(x):
    return (fast_tanh(x)+1)/2

def tanh(x):
    import theano.tensor as T
    return T.tanh(x)

def sigmoid(x):
    return tanh(x)/2 + 0.5


