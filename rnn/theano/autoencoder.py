
import theano
import theano.tensor as T

class NullNoise(object):
    def __call__(self, x, **kwargs):
        return x

    @property
    def weights(self):
        return []

class UniformNoise(object):
    def __init__(self, dim, p=0.25, seed=None):
        self.dim = dim
        self.p = p
        self.seed = seed

    def __call__(self, x, **kwargs):
        if self.p > 0:
            from theano.tensor.shared_randomstreams import RandomStreams
            if x.dtype.startswith('int'):
                #make one hot version of x
                shape = list(x.shape)+[self.dim]
                mesh = T.mgrid[0:x.shape[0],0:x.shape[1]]
                i,j = mesh[0], mesh[1]
                x_one_hot = T.set_subtensor(T.zeros(shape)[i,j,x], 1)
                x = x_one_hot
            rng = RandomStreams(seed=self.seed)
            I = rng.uniform((x.shape[0],x.shape[1])) < self.p
            x = T.set_subtensor(x[I.nonzero()], 1.0/self.dim)
        return x

    @property
    def weights(self):
        return []

class Autoencoder(object):
    def __init__(self, encoder, decoder, noise=NullNoise()):
        self.encoder = encoder
        self.decoder = decoder
        self.noise = noise

    @property
    def weights(self):
        return self.encoder.weights + self.decoder.weights + self.noise.weights

    def loss(self, X, mask, **kwargs):
        X_err = self.noise(X, mask=mask)
        Z = self.encoder(X_err, mask=mask, **kwargs)
        return self.decoder.loss(Z, X, mask=mask, **kwargs)

    def gradient(self, X, mask, **kwargs):
        res = self.loss(X, mask, **kwargs)
        L = T.sum(res[0])
        res = [T.sum(r, axis=[0,1]) for r in res]
        return theano.grad(L, self.weights), res 

    def transform(self, X, mask):
        return self.encoder(X, mask=mask)






