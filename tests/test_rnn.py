import unittest
import mindspore
import numpy as np
from mindspore import Tensor
from elmo.nn.rnn import DynamicRNN
from elmo.nn.rnn_cells import LSTMCell, LSTMCellWithProjection
from mindspore import context

class TestRNN(unittest.TestCase):
    def test_dynamic_rnn(self):
        context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')
        inputs = Tensor(np.random.randn(10, 3, 10), mindspore.float32)
        hx = Tensor(np.random.randn(3, 20), mindspore.float32)
        cx = Tensor(np.random.randn(3, 20), mindspore.float32)
        cell = LSTMCell(10, 20)
        rnn = DynamicRNN(cell)

        outputs, (hy, cy) = rnn(inputs, (hx, cx))

        assert outputs.shape == (10, 3, 20)
        assert hy.shape == (3, 20)
        assert cy.shape == (3, 20)
         
    def test_dynamic_rnn_with_projection(self):
        context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')
        inputs = Tensor(np.random.randn(10, 3, 10), mindspore.float32)
        hx = Tensor(np.random.randn(3, 30), mindspore.float32)
        cx = Tensor(np.random.randn(3, 20), mindspore.float32)
        cell = LSTMCellWithProjection(10, 20, True, 1, 30, 1)
        rnn = DynamicRNN(cell)

        outputs, (hy, cy) = rnn(inputs, (hx, cx))

        assert outputs.shape == (10, 3, 30)
        assert hy.shape == (3, 30)
        assert cy.shape == (3, 20)