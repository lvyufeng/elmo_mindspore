import unittest
import mindspore
import numpy as np
from mindspore import Tensor
from elmo.modules.embedding import CharacterEncoder
from mindspore import context

class TestCharEncoder(unittest.TestCase):
    char_cnn = {'activation': 'relu',
            'embedding': {'dim': 16},
            'filters': [[1, 32],
            [2, 32],
            [3, 64],
            [4, 128],
            [5, 256],
            [6, 512],
            [7, 1024]],
            'max_characters_per_token': 50,
            'n_characters': 261,
            'n_highway': 2}

    def test_char_encoder(self):
        context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')
        cnn_options = self.char_cnn
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)
        max_chars = cnn_options['max_characters_per_token']
        char_embed_dim = cnn_options['embedding']['dim']
        n_chars = cnn_options['n_characters']
        activation = cnn_options['activation']
        n_highway = cnn_options.get('n_highway')
        projection_dim = 512
        char_embedding = char_embedding = CharacterEncoder(filters, n_filters, max_chars, char_embed_dim, n_chars, n_highway, projection_dim, activation)

        # (batch_size, sequence_length, max_chars)
        inputs = Tensor(np.random.randn(3, 20, max_chars), mindspore.int32)

        token_embedding = char_embedding(inputs)
        # (num_layers, seq_length, batch_size, hidden_size)
        assert token_embedding.shape == (3, 20, projection_dim)

    def test_char_encoder_graph_mode(self):
        cnn_options = self.char_cnn
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)
        max_chars = cnn_options['max_characters_per_token']
        char_embed_dim = cnn_options['embedding']['dim']
        n_chars = cnn_options['n_characters']
        activation = cnn_options['activation']
        n_highway = cnn_options.get('n_highway')
        projection_dim = 512

        context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')

        char_embedding = char_embedding = CharacterEncoder(filters, n_filters, max_chars, char_embed_dim, n_chars, n_highway, projection_dim, activation)

        # (batch_size, sequence_length, max_chars)
        inputs = Tensor(np.random.randn(3, 20, max_chars), mindspore.int32)

        token_embedding = char_embedding(inputs)
        # (num_layers, seq_length, batch_size, hidden_size)
        assert token_embedding.shape == (3, 20, projection_dim)
        