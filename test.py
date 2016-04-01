import theano
import theano.tensor as T
import numpy as np

from rnn.theano.linear import Linear

lin = Linear(5, 10)
x = T.tensor3()
y = lin(x)
f = theano.function([x], y)
print 'Compiled.'
k,b = 100, 10
x = np.random.randn(k, b, 5)
print f(x).shape
