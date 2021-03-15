import mindspore
import mindspore.nn as nn
import mindspore.ops as P
from elmo.modules.embedding import CharacterEncoder
from elmo.modules.lstm import ELMoLSTM
from elmo.modules.loss import LossCell

class LanguageModel(nn.Cell):
    def __init__(self, options, training):
        """
        structure:
            embedding
            bilstm
            softmax
        """
        super().__init__()
        self.options = options
        n_tokens_vocab = self.options['n_tokens_vocab']
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']
        sample_softmax = self.options.get('sample_softmax', True)
        n_negative_samples_batch = self.options['n_negative_samples_batch']
        # LSTM options
        lstm_dim = self.options['lstm']['dim']
        projection_dim = self.options['lstm']['projection_dim']
        n_lstm_layers = self.options['lstm'].get('n_layers', 1)
        dropout = self.options['dropout']
        keep_prob = 1.0 - dropout

        cell_clip = self.options['lstm'].get('cell_clip')
        proj_clip = self.options['lstm'].get('proj_clip')

        use_skip_connections = self.options['lstm'].get(
                                            'use_skip_connections')
        # CNN options
        cnn_options = self.options['char_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)
        max_chars = cnn_options['max_characters_per_token']
        char_embed_dim = cnn_options['embedding']['dim']
        n_chars = cnn_options['n_characters']
        activation = cnn_options['activation']
        n_highway = cnn_options.get('n_highway')
        
        self.char_embedding = CharacterEncoder(filters, n_filters, max_chars, char_embed_dim, n_chars, n_highway, projection_dim, activation)
        self.bilstm = ELMoLSTM(projection_dim, lstm_dim, projection_dim, n_lstm_layers, keep_prob, cell_clip, proj_clip, use_skip_connections, is_training=True, batch_first=True)

        self.loss = LossCell(projection_dim, n_tokens_vocab, sample_softmax, n_negative_samples_batch, training=training)
    
    def construct(self, inputs, next_ids_forward, next_ids_backward):
        """
            args:
                inputs: (batch_size, sequence_length, max_chars)
                next_ids_forward: (batch_size, sequence_length)
                next_ids_backward: (batch_size, sequence_length)
        """
        # (batch_size, sequence_length, embedding_dim)
        token_embedding = self.char_embedding(inputs)
        # (num_layers, batch_size, sequence_length, embedding_dim)
        encoder_output, _ = self.bilstm(token_embedding)
        # (batch_size, sequence_length, embedding_dim * num_directions)
        encoder_output = encoder_output[1]
        # (batch_size, sequence_length, embedding_dim)
        forward, backward = P.Split(2, 2)(encoder_output)
        
        loss = self.loss((forward, backward), (next_ids_forward, next_ids_backward))
        return loss