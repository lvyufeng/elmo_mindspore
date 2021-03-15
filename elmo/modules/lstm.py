import mindspore
import mindspore.nn as nn
import mindspore.ops as P
import numpy as np
from elmo.nn.rnn import DynamicRNN
from elmo.nn.rnn_cells import LSTMCellWithProjection
from elmo.nn.rnn_cell_wrapper import ResidualWrapper, DropoutWrapper
from mindspore.ops.primitive import constexpr
from mindspore import Tensor

@constexpr
def _init_state(hidden_num, batch_size, hidden_size, proj_size, dtype):
    hx = Tensor(np.zeros((hidden_num, batch_size, proj_size)), dtype)
    cx = Tensor(np.zeros((hidden_num, batch_size, hidden_size)), dtype)
    return (hx, cx)
    
class ELMoLSTM(nn.Cell):
    def __init__(
                self, 
                input_size, 
                hidden_size, 
                proj_size,
                num_layers, 
                keep_prob:float=0.0,
                cell_clip:float=0.0,
                proj_clip:float=0.0,
                skip_connections:bool=False,
                is_training:bool=True,
                batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.proj_size = proj_size
        self.num_directions = 2
        self.batch_first = batch_first

        forward_layers = []
        backward_layers = []
        lstm_input_size = input_size
        for i in range(num_layers):
            forward_cell = LSTMCellWithProjection(lstm_input_size, hidden_size, cell_clip=cell_clip, proj_size=proj_size, proj_clip=proj_clip)            
            backward_cell = LSTMCellWithProjection(lstm_input_size, hidden_size, cell_clip=cell_clip, proj_size=proj_size, proj_clip=proj_clip)
            
            if skip_connections:
                if i == 0:
                    pass
                else:
                    forward_cell = ResidualWrapper(forward_cell)
                    backward_cell = ResidualWrapper(backward_cell)
            
            if is_training:
                forward_cell = DropoutWrapper(forward_cell, input_keep_prob=keep_prob)
                backward_cell = DropoutWrapper(backward_cell, input_keep_prob=keep_prob)

            forward_layer = DynamicRNN(forward_cell)
            backward_layer = DynamicRNN(backward_cell)
            
            lstm_input_size = proj_size
            forward_layers.append(forward_layer)
            backward_layers.append(backward_layer)

        self.forward_layers = forward_layers
        self.backward_layers = backward_layers
        self.dropout = nn.Dropout(keep_prob=keep_prob)

    def construct(self, x, h=None, seq_length=None):
        max_batch_size = x.shape[0] if self.batch_first else x.shape[1]
        if h is None:
            h = _init_state(self.num_layers * self.num_directions, max_batch_size, self.hidden_size, self.proj_size, x.dtype)
        if self.batch_first:
            x = P.Transpose()(x, (1, 0, 2))
        x, h = self._stacked_bi_dynamic_rnn(x, h, seq_length)
        if self.batch_first:
           x = P.Transpose()(x, (0, 2, 1, 3))
        return x, h

    def _stacked_bi_dynamic_rnn(self, x, h, seq_length):
        """stacked bidirectional dynamic_rnn"""
        input_forward = x
        if seq_length is None:
            input_backward = P.ReverseV2([0])(input_forward)
        else:
            input_backward = P.ReverseSequence(0, 1)(input_forward, seq_length)
        h_n = ()
        c_n = ()
        outputs = ()
        for i, (forward_cell, backward_cell) in enumerate(zip(self.forward_layers, self.backward_layers)):
            offset = i * 2
            h_f_i = (h[0][offset], h[1][offset])
            h_b_i = (h[0][offset + 1], h[1][offset + 1])
 
            output_f, h_t_f = forward_cell(input_forward, h_f_i, seq_length)
            output_b, h_t_b = backward_cell(input_backward, h_b_i, seq_length)
            if seq_length is None:
                output_b = P.ReverseV2([0])(output_b)
            else:
                output_b = P.ReverseSequence(0, 1)(output_b, seq_length)

            output = P.Concat(2)((output_f, output_b))
            outputs += (output,)
            input_forward = output_f
            input_backward = output_b

            h_t = P.Concat(1)((h_t_f[0], h_t_b[0]))
            c_t = P.Concat(1)((h_t_f[1], h_t_b[1]))
            h_n += (h_t,)
            c_n += (c_t,)


        h_n = P.Stack(0)(h_n)
        c_n = P.Stack(0)(c_n)
        outputs = P.Stack(0)(outputs)
        outputs = self.dropout(outputs)

        return outputs, (h_n, c_n)