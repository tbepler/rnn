import theano
import theano.tensor as T
import numpy as np

from rnn.theano.autoencoder import Autoencoder, NullNoise

class PositionalAutoencoder(object):
    def __init__(self, encoder, decoder, noise=NullNoise(), **kwargs):
        super(PositionalAutoencoder, self).__init__(Autoencoder(encoder, decoder, noise=noise), **kwargs)

        
