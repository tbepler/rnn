import theano
import theano.tensor as T
import numpy as np

from rnn.models.context_embedding import ContextEmbedding
import rnn.theano.blstm_embedding as blstm_embedding

class BlstmEmbed(ContextEmbedding):
    def __init__(self, n_in, n_components, hidden_units=[], l2_reg=0.01, type='real', grad_clip=None, **kwargs):
        model = blstm_embedding.BlstmEmbed(n_in, n_components
                    , hidden_units=hidden_units
                    , l2_reg=l2_reg
                    , type=type
                    , grad_clip=grad_clip)
        super(BlstmEmbed, self).__init__(model, **kwargs)

        
