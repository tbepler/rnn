import theano
import theano.tensor as T
import numpy as np

from rnn.models.context_embedding import ContextEmbedding
from rnn.theano.autoencoder import NullNoise
import rnn.theano.topic_lstm as topic_lstm

class TopicLSTM(ContextEmbedding):
    def __init__(self, n_in, units, n_topics, sparsity=0, noise=NullNoise(), **kwargs):
        model = topic_lstm.TopicLSTM(n_in, units, n_topics, sparsity=sparsity, noise=noise)
        super(TopicLSTM, self).__init__(model, **kwargs)

        
