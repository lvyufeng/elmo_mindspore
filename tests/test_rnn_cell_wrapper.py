import unittest
import mindspore
import numpy as np
from mindspore import Tensor
from elmo.nn.rnn_cells import LSTMCell
from elmo.nn.rnn_cell_wrapper import DropoutWrapper, ResidualWrapper, MultiRNNCell

class TestRNNCellWrapper(unittest.TestCase):
    def test_dropout_wrapper(self):
        inputs = Tensor(np.random.randn(1, 10), mindspore.float32)
        hx = Tensor(np.random.randn(1, 20), mindspore.float32)
        cx = Tensor(np.random.randn(1, 20), mindspore.float32)
        cell = LSTMCell(10, 20)
        cell = DropoutWrapper(cell, input_keep_prob=0.5)
        hy, cy = cell(inputs, (hx, cx))

        assert hy.shape[-1] == 20

    def test_residual_wrapper(self):
        inputs = Tensor(np.random.randn(1, 20), mindspore.float32)
        hx = Tensor(np.random.randn(1, 20), mindspore.float32)
        cx = Tensor(np.random.randn(1, 20), mindspore.float32)
        cell = LSTMCell(20, 20)
        cell = ResidualWrapper(cell)
        hy, cy = cell(inputs, (hx, cx))

        assert hy.shape[-1] == 20