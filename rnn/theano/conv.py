from __future__ import print_function, division

import numpy as np
import theano
import theano.tensor as T
import theano.tensor.signal

from rnn.initializers import orthogonal, xavier

class Conv1d(object):
    def __init__(self, n_in, filters, width, stride=1, border_mode='half'
                , dilation=1
                , dtype=theano.config.floatX, random=np.random
                , use_bias=True):
        self.n_in = n_in
        self.filters = filters
        self.width = width
        self.stride = stride
        if border_mode == 'same':
            border_moe = 'half'
        self.border_mode = border_mode
        self.dilation = dilation
        w = random.randn(*self.shape).astype(dtype)
        xavier(w)
        self.w = theano.shared(w)
        if use_bias:
            b = np.zeros(filters, dtype=dtype)
            self.b = theano.shared(b)

    @property
    def shape(self):
        return (self.filters, self.n_in, self.width, 1)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['w'] = state['w'].get_value()
        if 'b' in state:
            state['b'] = state['b'].get_value()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.w = theano.shared(self.w)
        if hasattr(self, 'b'):
            self.b = theano.shared(self.b)

    @property
    def shared(self):
        if hasattr(self, 'b'):
            return [self.w, self.b]
        return [self.w]

    @property
    def weights(self):
        return [self.w]

    @property
    def bias(self):
        if hasattr(self, 'b'):
            return [self.b]
        return []

    def __call__(self, x, mask=None):
        if mask is not None:
            assert self.border_mode == 'half'
            x *= T.shape_padright(mask)

        ## x is (batch,length,features), need to reshape for convolution
        x = x.dimshuffle(0, 2, 1, 'x')
        y = T.nnet.conv2d(x, self.w, filter_shape=self.shape, border_mode=self.border_mode
                         , subsample=(self.stride, 1), filter_dilation=(self.dilation, 1))
        if hasattr(self, 'b'):
            y += self.b.dimshuffle('x', 0, 'x', 'x')
        ## shape y back
        return y[...,0].dimshuffle(0, 2, 1)


class GatedConv1d(Conv1d):
    def __init__(self, n_in, filters, width, activation=T.tanh, gate=T.nnet.sigmoid, **kwargs):
        super(GatedConv1d, self).__init__(n_in, 2*filters, width, **kwargs)
        self.activation = activation
        self.gate = gate

    def __call__(self, x, **kwargs):
        y = super(GatedConv1d, self).__call__(x, **kwargs)
        n = self.filters//2
        h = self.activation(y[...,0:n])
        g = self.gate(y[...,n:])
        return h*g




