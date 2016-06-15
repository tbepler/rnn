import theano
import theano.tensor as T
import numpy as np

from rnn.models.context_embedding import ContextEmbedding
import rnn.theano.coupling_lstm as coupling_lstm

class CouplingLSTM(ContextEmbedding):
    def __init__(self, n_in, n_components, layers=[], weights_r2=0.01, grad_clip=None, **kwargs):
        model = coupling_lstm.CouplingLSTM(n_in, n_topics, layers=layers, weights_r2=weights_r2, grad_clip=grad_clip)
        super(CouplingLSTM, self).__init__(model, **kwargs)

        
