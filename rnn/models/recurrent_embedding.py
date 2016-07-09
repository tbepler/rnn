import theano
import theano.tensor as T
import numpy as np

import rnn.theano.solvers as solvers
from rnn.models.context_embedding import ContextEmbedding
import rnn.theano.recurrent_embedding as recurrent_embedding

class RecurrentEmbed(ContextEmbedding):
    def __init__(self, encoder, decoder, optimizer=solvers.RMSprop(0.01), batch_size=100, length=-1, flank=0
            , **kwargs):
        model = recurrent_embedding.RecurrentEmbed(encoder, decoder, **kwargs)
        super(RecurrentEmbed, self).__init__(model, optimizer=optimizer, batch_size=batch_size, length=length
                , flank=flank)

        
