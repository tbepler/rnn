import theano
import theano.tensor as T
import numpy as np

from rnn.models.context_embedding import ContextEmbedding, null_func
import rnn.theano.coupling_lstm as coupling_lstm

class CouplingLSTM(ContextEmbedding):
    def __init__(self, n_in, n_components, layers=[], atten_layers=[], weights_r2=0.01, grad_clip=None, **kwargs):
        model = coupling_lstm.CouplingLSTM(n_in, n_components, layers=layers, atten_layers=atten_layers, weights_r2=weights_r2, grad_clip=grad_clip)
        super(CouplingLSTM, self).__init__(model, **kwargs)

    def _compile_attention_minibatch(self, dtype):
        if not hasattr(self, 'attention_minibatch'):
            flank = T.iscalar()
            X = T.matrix(dtype=dtype)
            mask = self._theano_mask()
            n = X.shape[0]
            Z = self.encoder.attention(X, mask)
            Z = Z[flank:n-flank]
            return theano.function([X, mask, flank], Z)  
        else:
            return self.attention_minibatch

    def attention(self, data, callback=null_func, **kwargs):
        attention_minibatch = self._compile_attention_minibatch(data[0].dtype)
        total = 0
        if hasattr(data, '__len__'):
            total = sum(len(x) for x in data)
            callback(0, 'attention')
        n = 0
        for index,X,M in self.minibatched(self.split(data)):
            Z = attention_minibatch(X, M, self.flank)
            n += sum(idx[2] for idx in index)
            yield index, Z
            if total > 0:
                callback(float(n)/total, 'attention')


        
