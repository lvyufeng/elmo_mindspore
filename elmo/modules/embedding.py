import json
import mindspore
import mindspore.nn as nn
import numpy as np
import mindspore.ops as P
from mindspore.common.initializer import initializer, Uniform
from elmo.modules.highway import HighWay
from elmo.nn.layers import Conv1d, Dense, Embedding

class CharacterEncoder(nn.Cell):
    """
    Compute context sensitive token representation using pretrained biLM.

    This embedder has input character ids of size (batch_size, sequence_length, 50)
    and returns (batch_size, sequence_length + 2, embedding_dim), where embedding_dim
    is specified in the options file (typically 512).

    We add special entries at the beginning and end of each sequence corresponding
    to <S> and </S>, the beginning and end of sentence tokens.
    """
    def __init__(self, options_file:str):
        super().__init__()
        with open(options_file, 'r') as f:
            self._options = json.load(f)
        cnn_options = self._options['char_cnn']        
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)
        max_chars = cnn_options['max_characters_per_token']
        char_embed_dim = cnn_options['embedding']['dim']
        n_chars = cnn_options['n_characters']
        n_highway = cnn_options['n_highway']
        output_dim = self._options['lstm']['projection_dim']

        self.max_chars_per_token = self._options['char_cnn']['max_characters_per_token']

        # init char_embedding
        self.char_embedding = Embedding(n_chars + 1, char_embed_dim, embedding_table=Uniform(1.0), padding_idx=0)
        # run convolutions
        convolutions = []
        for i, (width, num) in enumerate(filters):
            conv = Conv1d(
                in_channels=char_embed_dim,
                out_channels=num,
                kernel_size=width,
                has_bias=True
            )
        self._convolutions = convolutions
        # activation for convolutions
        cnn_options = self._options['char_cnn']
        if cnn_options['activation'] == 'tanh':
            self._activation = nn.Tanh()
        elif cnn_options['activation'] == 'relu':
            self._activation = nn.ReLU()
        else:
            raise ValueError("Unknown activation")
        # highway layers
        self._highways = HighWay(n_filters, n_highway, 'relu')
        # projection layer
        self._projection = Dense(n_filters, output_dim, has_bias=True)
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
        batch_size, sequence_length, _ = inputs.shape()
        return token_embedding.view(batch_size, sequence_length, -1)
