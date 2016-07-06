import theano.tensor as T

def fast_tanh(x):
    return x/(1+abs(x))

def fast_sigmoid(x):
    return (fast_tanh(x)+1)/2

def tanh(x):
    return T.tanh(x)

def sigmoid(x):
    return tanh(x)/2 + 0.5

def normalize(x, axis=-1):
    Z = T.sqrt(T.sum(x**2, axis=axis, keepdims=True))
    #the derivative of 1/sqrt(sum(x**2)) has sqrt(sum(x**2))**3 in the denominator
    #meaning thate sqrt(sum(x**2)) needs to be truncated at a larger value to
    #prevent sqrt(sum(x**2))**3 from being zero
    Z = T.maximum(Z, 1e-12) 
    return x/Z

def dampen(x):
    #transform x as sign(x)*ln(|x|+1)
    # preserves full range, but dampens the gradient
    return T.sgn(x)*T.log1p(abs(x))

