import numpy as np

def compose(*args):
    x = args[0]
    for i in xrange(1, len(args)):
        x = Compose(x, args[i])
    return x

#this layer composes two sub layers.
# ie) g(f(x))
class Compose(object):

    def __init__(self, f, g):
        self.f = f
        self.g = g

    def weights(self, dtype=np.float64):
        Wf = self.f.weights(dtype=dtype).reshape(self.f.size)
        Wg = self.g.weights(dtype=dtype).reshape(self.g.size)
        return np.concatenate((Wf, Wg))
        
    @property
    def shape(self):
        return self.size

    @property
    def size(self):
        return self.f.size + self.g.size

    @property
    def Y(self):
        return self.g.Y

    @property
    def dX(self):
        return self.f.dX

    @property
    def inputs(self):
        return self.f.inputs

    @property
    def outputs(self):
        return self.g.outputs

    def reset(self):
        self.f.reset()
        self.g.reset()

    def advance(self):
        self.f.advance()
        self.g.advance()
            
    def forward(self, W, X, train=None):
        Wf = W[:self.f.size].reshape(self.f.shape)
        Yf = self.f.forward(Wf, X, train=train)
        Wg = W[self.f.size:].reshape(self.g.shape)
        Yg = self.g.forward(Wg, Yf, train=train)
        self.X = X
        return Yg
    
    def backward(self, W, dY, dW):
        Wg = W[self.f.size:].reshape(self.g.shape)
        dWg = dW[self.f.size:].reshape(self.g.shape)
        dYf = self.g.backward(Wg, dY, dWg)
        Wf = W[:self.f.size].reshape(self.f.shape)
        dWf = dW[:self.f.size].reshape(self.f.shape)
        dX = self.f.backward(Wf, dYf, dWf)
        return dX
        
if __name__ == '__main__':
    import test
    import linear
    import lstm
    import lstm_encoder
    import softmax
    import copy
    ins = 4
    outs = 4

    layers = [linear.Linear(ins,outs), lstm.LSTM(ins,outs)
              , lstm_encoder.LSTMEncoder(ins,outs)]
    for f in layers:
        for g in layers:
            layer = Compose(copy.deepcopy(f), copy.deepcopy(g))
            print "Testing composition:", type(f), type(g)
            test.layer(layer, float_t=np.float64)
            test.layer(layer, float_t=np.float32)

    layer = Compose(lstm_encoder.LSTMEncoder(ins,outs), softmax.SoftmaxCrossEntropy())
    print "Testing composition:", type(layer.f), type(layer.g)
    
    error_f = softmax.CrossEntropyError(np.int32)
    test.layer(layer, outs=outs, err_f=error_f, float_t = np.float64)
    test.layer(layer, outs=outs, err_f=error_f, float_t = np.float32)

    error_f = softmax.CrossEntropyError(np.int64)
    test.layer(layer, outs=outs, err_f=error_f, float_t=np.float64)
    test.layer(layer, outs=outs, err_f=error_f, float_t=np.float32)
       
