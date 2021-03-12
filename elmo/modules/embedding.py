import json
import mindspore
import mindspore.nn as nn
import numpy as np
import mindspore.ops as P
from mindspore.common.initializer import Uniform, Normal
from elmo.modules.highway import HighWay
from elmo.nn.layers import Conv1d, Dense, Embedding

"""
Notice: We don't use `bidirectional` flag to encode the input sequences
on both side. To bidirectional usage, just initalize another Encoder instance.
"""

class CharacterEncoder(nn.Cell):
    """
    Compute context sensitive token representation using pretrained biLM.

    This embedder has input character ids of size (batch_size, sequence_length, 50)
    and returns (batch_size, sequence_length + 2, embedding_dim), where embedding_dim
    is specified in the options file (typically 512).

    We add special entries at the beginning and end of each sequence corresponding
    to <S> and </S>, the beginning and end of sentence tokens.
    """
    def __init__(self, 
                filters,
                n_filters,
                max_chars_per_token,
                char_embed_dim,
                n_chars,
                n_highway,
                output_dim,
                activation):
        super().__init__()

        self.max_chars_per_token = max_chars_per_token

        # activation for convolutions
        if activation == 'tanh':
            self._activation = nn.Tanh()
        elif activation == 'relu':
            self._activation = nn.ReLU()
        else:
            raise ValueError("Unknown activation")

        # init char_embedding
        self.char_embedding = Embedding(n_chars + 1, char_embed_dim, embedding_table=Uniform(1.0), padding_idx=0)
        # run convolutions
        convolutions = []
        for (width, num) in filters:
            if activation == 'tanh':
                cnn_weight_init = Normal(np.sqrt(1.0 / width * char_embed_dim))
            elif activation == 'relu':
                cnn_weight_init = Uniform(0.05)
            conv = nn.Conv1d(
                in_channels=char_embed_dim,
                out_channels=num,
                kernel_size=width,
                has_bias=True,
                weight_init=cnn_weight_init
            )
            convolutions.append(conv)
        self._convolutions = convolutions

        # highway layers
        self._highways = HighWay(n_filters, n_highway, 'relu')
        # projection layer
        self._projection = nn.Dense(n_filters, output_dim, has_bias=True, weight_init=Normal(np.sqrt(1.0 / n_filters)))
        # array operations
        self.transpose = P.Transpose()
        self.concat = P.Concat(-1)
        self.max = P.ReduceMax()

    def construct(self, inputs):
        # (batch_size * sequence_length, max_chars_per_token, embed_dim)
        character_embedding = self.char_embedding(inputs.view(-1, self.max_chars_per_token))

        # (batch_size * sequence_length, embed_dim, max_chars_per_token)
        character_embedding = self.transpose(character_embedding, (0, 2, 1))
        convs = ()
        for conv in self._convolutions:
            convolved = conv(character_embedding)
            # (batch_size * sequence_length, n_filters for this width)
            convolved = self.max(convolved, axis=-1)
            convolved = self._activation(convolved)
            convs += (convolved, )
        
        # (batch_size * sequence_length, n_filters)
        token_embedding = self.concat(convolved)

        # apply the highway layers (batch_size * sequence_length, n_filters)
        token_embedding = self._highways(token_embedding)

        # final projection (batch_size * sequence_length, embedding_dim)
        token_embedding = self._projection(token_embedding)

        # reshape to (batch_size, sequence_length, embedding_dim)
        batch_size, sequence_length, _ = inputs.shape
        return token_embedding.view(batch_size, sequence_length, -1)
