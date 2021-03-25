import math
import mindspore
import numpy as np
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, Uniform
from elmo.utils.glorot_uniform import glorot_uniform
from typing import Tuple

def rnn_tanh_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    if b_ih is None:
        igates = P.MatMul(False, True)(input, w_ih)
        hgates = P.MatMul(False, True)(hidden, w_hh)
    else:
        igates = P.MatMul(False, True)(input, w_ih) + b_ih
        hgates = P.MatMul(False, True)(hidden, w_hh) + b_hh
    return P.Tanh()(igates + hgates)

def rnn_relu_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    if b_ih is None:
        igates = P.MatMul(False, True)(input, w_ih)
        hgates = P.MatMul(False, True)(hidden, w_hh)
    else:
        igates = P.MatMul(False, True)(input, w_ih) + b_ih
        hgates = P.MatMul(False, True)(hidden, w_hh) + b_hh
    return P.ReLU()(igates + hgates)

def lstm_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    hx, cx = hidden
    if b_ih is None:
        gates = P.MatMul(False, True)(input, w_ih) + P.MatMul(False, True)(hx, w_hh)
    else:
        gates = P.MatMul(False, True)(input, w_ih) + P.MatMul(False, True)(hx, w_hh) + b_ih + b_hh
    ingate, forgetgate, cellgate, outgate = P.Split(1, 4)(gates)
    
    ingate = P.Sigmoid()(ingate)
    forgetgate = P.Sigmoid()(forgetgate)
    cellgate = P.Tanh()(cellgate)
    outgate = P.Sigmoid()(outgate)
    
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * P.Tanh()(cy)
    
    return hy, cy

def gru_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    if b_ih is None:
        gi = P.MatMul(False, True)(input, w_ih)
        gh = P.MatMul(False, True)(hidden, w_hh)
    else:
        gi = P.MatMul(False, True)(input, w_ih) + b_ih
        gh = P.MatMul(False, True)(hidden, w_hh) + b_hh
    i_r, i_i, i_n = P.Split(1, 3)(gi)
    h_r, h_i, h_n = P.Split(1, 3)(gh)
    
    resetgate = P.Sigmoid()(i_r + h_r)
    inputgate = P.Sigmoid()(i_i + h_i)
    newgate = P.Tanh()(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)
    
    return hy


class RNNCellBase(nn.Cell):
    def __init__(self, input_size: int, cell_size: int, bias: bool, num_chunks: int, proj_size:int=None):
        super().__init__()
        self.input_size = input_size
        self.cell_size = cell_size
        self.bias = bias

        hidden_size = proj_size if proj_size else cell_size
        self.weight_ih = Parameter(glorot_uniform(num_chunks * cell_size, input_size))
        self.weight_hh = Parameter(glorot_uniform(num_chunks * cell_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(Tensor(np.zeros((num_chunks * cell_size)).astype(np.float32)))
            self.bias_hh = Parameter(Tensor(np.zeros((num_chunks * cell_size)).astype(np.float32)))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.cell_size)
        for weight in self.get_parameters():
            weight.set_data(initializer(Uniform(stdv), weight.shape))
            
class RNNCell(RNNCellBase):
    _non_linearity = ['tanh', 'relu']
    def __init__(self, input_size: int, hidden_size: int, bias: bool=True, nonlinearity: str = "tanh"):
        super().__init__(input_size, hidden_size, bias, num_chunks=1)
        if nonlinearity not in self._non_linearity:
            raise ValueError("Unknown nonlinearity: {}".format(nonlinearity))
        self.nonlinearity = nonlinearity
        self.cell_type = 'RNN'
    def construct(self, input, hx):
        if self.nonlinearity == "tanh":
            ret = rnn_tanh_cell(input, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        else:
            ret = rnn_relu_cell(input, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        return ret
    
class LSTMCell(RNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__(input_size, hidden_size, bias, num_chunks=4)
        self.support_non_tensor_inputs = True
        self.cell_type = 'LSTM'
    def construct(self, input, hx):
        return lstm_cell(input, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
    
class GRUCell(RNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__(input_size, hidden_size, bias, num_chunks=3)
        self.cell_type = 'GRU'
    def construct(self, input, hx):
        return gru_cell(input, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)

class LSTMCellWithProjection(RNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, \
        cell_clip=None, proj_size=None, proj_clip=None):
        super().__init__(input_size, hidden_size, bias, num_chunks=4, proj_size=proj_size)

        self.cell_clip = cell_clip
        self.proj_size = proj_size
        self.proj_clip = proj_clip
        if proj_size is not None:
            self.proj_weight = Parameter(Tensor(np.random.randn(hidden_size, proj_size).astype(np.float32)))

        self.matmul = P.MatMul()

        self.cell_type = 'LSTM'
    def construct(self, inputs, hx):
        hy, cy = lstm_cell(inputs, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        if self.cell_clip is not None:
            cy = P.clip_by_value(cy, -self.cell_clip, self.cell_clip)
        if self.proj_size is not None:
            hy = self.matmul(hy, self.proj_weight)
            if self.proj_clip is not None:
                hy = P.clip_by_value(hy, -self.proj_clip, self.proj_clip)

        return hy, cy