import mindspore
import mindspore.nn as nn
import mindspore.ops as P

class DynamicRNN(nn.Cell):
    def __init__(self, cell):
        super().__init__()
        self.cell = cell
        self.is_lstm = cell.cell_type == 'LSTM'
        
    def recurrent(self, x, h):
        time_step = range(x.shape[0])
        outputs = []
        for t in time_step:
            h = self.cell(x[t], h)
            if self.is_lstm:
                outputs.append(h[0])
            else:
                outputs.append(h)
        outputs = P.Stack()(outputs)
        return outputs, h
    
    def variable_recurrent(self, x, h, seq_length):
        time_step = range(x.shape[0])
        h_t = h
        if self.is_lstm:
            hidden_size = h[0].shape[-1]
            zero_output = P.ZerosLike()(h_t[0])
        else:
            hidden_size = h.shape[-1]
            zero_output = P.ZerosLike()(h_t)
        
        seq_length = P.BroadcastTo((hidden_size, -1))(seq_length)
        seq_length = P.Transpose()(seq_length, (1, 0))
        
        outputs = []
        state_t = h_t
        for t in time_step:
            h_t = self.cell(x[t], state_t)
            seq_cond = seq_length > t
            if self.is_lstm:
                state_t_0 = P.Select()(seq_cond, h_t[0], state_t[0])
                state_t_1 = P.Select()(seq_cond, h_t[1], state_t[1])
                output = P.Select()(seq_cond, h_t[0], zero_output)
                state_t = (state_t_0, state_t_1)
            else:
                state_t = P.Select()(seq_cond, h_t, state_t)
                output = P.Select()(seq_cond, h_t, zero_output)
            outputs.append(output)
        outputs = P.Stack()(outputs)
        return outputs, state_t
    
    def construct(self, x, h, seq_length=None):
        if seq_length is None:
            return self.recurrent(x, h)
        else:
            return self.variable_recurrent(x, h, seq_length)