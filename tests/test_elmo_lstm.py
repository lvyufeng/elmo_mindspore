import unittest
import mindspore
import numpy as np
from mindspore import Tensor
from elmo.modules.lstm import ELMoLSTM

class TestELMoLSTM(unittest.TestCase):
    def test_elmo_lstm(self):
        inputs = Tensor(np.random.randn(10, 3, 10), mindspore.float32)
        hx = Tensor(np.random.randn(4, 3, 30), mindspore.float32)
        cx = Tensor(np.random.randn(4, 3, 20), mindspore.float32)
        lstm = ELMoLSTM(10, 20, 30, 2, 0.5, 1.0, 1.0, True, True)

        outputs, (hy, cy) = lstm(inputs, (hx, cx))

        # (num_layers, seq_length, batch_size, hidden_size)
        assert outputs.shape == (2, 10, 3, 60)
        # (num_layers, batch_size, hidden_size)
        assert hy.shape == (2, 3, 60)
        # (num_layers, batch_size, hidden_size)
        assert cy.shape == (2, 3, 40)