import theano
import theano.tensor as T
import numpy as np

import rnn.theano.solvers as solvers
from rnn.models.context_embedding import ContextEmbedding
import rnn.theano.blstm_embedding as blstm_embedding

class BlstmEmbed(ContextEmbedding):
    def __init__(self, n_in, n_components, optimizer=solvers.RMSprop(0.01), batch_size=100, length=-1, flank=0
            , **kwargs):
        model = blstm_embedding.BlstmEmbed(n_in, n_components, **kwargs)
        super(BlstmEmbed, self).__init__(model, optimizer=optimizer, batch_size=batch_size, length=length
                , flank=flank)

        
