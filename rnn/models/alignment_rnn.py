
from rnn.theano.lstm import LayeredBLSTM, LayeredLSTM


class RecurrentAttentionDecoder(object):
    def __init__(self, n, h, units, forget_bias=3):
        from rnn.theano.initializers import orthogonal
        b = np.zeros(4*units, dtype=theano.config.floatX)
        b[units:2*units] = forget_bias
        self.b = theano.shared(b, borrow=True)
        Wy = np.random.randn(n, 4*units).astype(theano.config.floatX)
        orthogonal(Wy)
        self.Wy = theano.shared(Wy, borrow=True)

    def step(self, IFOG):
        pass

    def __call__(self, H, Y):
        import rnn.theano.lstm as lstm
        IFOG = lstm.gates(self.b, self.Wy, Y)



class AlignmentRNN(object):
    def __init__(self, n, encoder_layers, decoder_layers, l2_reg=0.01):
        self.encoder = LayeredBLSTM(n, encoder_layers)

        pass


