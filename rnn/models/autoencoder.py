import theano
import theano.tensor as T
import numpy as np

class PositionalAutoencoder(object):
    def __init__(self, encoder, decoder, error_model):
        self.encoder = encoder
        self.decoder = decoder
        self.error_model = error_model

    @property
    def weights(self):
        return self.encoder.weights + self.decoder.weights + self.error_model.weights

    def loss(self, X, mask):
        X_err = self.error_model(X, mask=mask)
        Z = self.encoder(X_err, mask=mask)
        return self.decoder.loss(X_err, X, mask=mask)

    def gradient(self, X, mask, flank=0):
        res = self.loss(X, mask)
        Ls = []
        for r in res:
            n = r.shape[0]
            Ls.append(T.sum(r[flank:n-flank], axis=[0,1]))
        L = T.sum(res[0][flank:res[0].shape[0]-flank])
        return theano.grad(L, self.weights), Ls 

    def transform(self, X, mask):
        return self.encoder(X, mask=mask)



        
