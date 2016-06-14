import theano
import theano.tensor as T
import numpy as np

from rnn.models.context_embedding import ContextEmbedding
from rnn.theano.autoencoder import NullNoise
import rnn.theano.topic_lstm_v2 as topic_lstm

class TopicLSTM(ContextEmbedding):
    def __init__(self, n_in, n_topics, n_components, lstm_layers=[]
            , sparsity=0, unscaled_topic_r2=0, weights_r2=0.01, topic_orth_r2 = 0
            , grad_clip=1.0, **kwargs):
        model = topic_lstm.TopicLSTM(n_in, n_topics, n_components, lstm_layers=lstm_layers
                 , sparsity=sparsity, unscaled_topic_r2=unscaled_topic_r2, weights_r2=weights_r2
                 , topic_orth_r2=topic_orth_r2, grad_clip=grad_clip)
        super(TopicLSTM, self).__init__(model, **kwargs)

        
