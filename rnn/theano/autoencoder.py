
import theano
import theano.tensor as T

import rnn.theano.lstm as lstm
import rnn.theano.crf as crf
import rnn.theano.linear as linear
import rnn.theano.softmax as softmax
import rnn.theano.crossent as crossent

class Encoder(object):
    def __init__(self, n_in, dims, layers):
        self.layers = []
        m = n_in
        for n in layers:
            L = lstm.BiLSTM(m, n)
            self.layers.append(L)
            m = n
        self.decoder = lstm.BiLSTM(m, dims)

    @property
    def weights(self):
        return [layer.weights for layer in self.layers] + [self.decoder.weights]

    @weights.setter
    def weights(self, weights):
        for i in xrange(len(self.layers)):
            self.layers[i].weights.set_value(weights[i])
        self.decoder.weights.set_value(weights[-1])

    def __call__(self, X, mask=None):
        for L in self.layers:
            X = L.scan(X, mask=mask)[0]
        return self.decoder.fold(X, mask=mask)[0]

class Decoder(object):
    def __init__(self, dims, n_out, layers=[]):
        if len(layers) == 0:
            layers = [2*n_out]
        self.layers = []
        m = dims
        for n in layers:
            L = lstmBiLSTM(m, n)
            self.layers.append(L)
            m = n
        self.decoder = crf.CRF(m, n_out)

    @property
    def weights(self):
        return [layer.weights for layer in self.layers] + self.decoder.weights

    @weights.setter
    def weights(self, ws):
        for i in xrange(len(self.layers)):
            self.layers[i].weights.set_value(ws[i])
        self.decoder.weights = ws[-1]
        
    def logprob(self, X, Y, mask=None):
        n = Y.shape[0]
        X = self.layers[0].unfold(X, n, mask=mask)[0]
        for L in self.layers[1:]:
            X = L.scan(X, mask=mask)[0]
        return self.decoder.logprob(Y, X, mask=mask)

class Autoencoder(object):
    def __init__(self, n_in, dims, layers, encoder=None, decoder=None):
        if encoder is None:
            encoder = Encoder(n_in, dims, layers)
        if decoder is None:
            decoder = Decoder(dims, n_in)
        self.encoder = encoder
        self.decoder = decoder

    @property
    def weights(self):
        return self.encoder.weights + self.decoder.weights

    @weights.setter
    def weights(self, weights):
        n = len(self.encoder.weights)
        self.encoder.weights = weights[:n]
        self.decoder.weights = weights[n:]

    def transform(self, X, mask=None):
        return self.encoder(X, mask=mask)

    def loss(self, X, mask=None):
        Z = self.transform(X, mask=mask)
        return -self.decoder.logprob(Z, X, mask=mask)







