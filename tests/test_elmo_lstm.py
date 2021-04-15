import unittest
import mindspore
import numpy as np
from mindspore import Tensor
from elmo.modules.lstm import ELMoLSTM
from mindspore import context

class TestELMoLSTM(unittest.TestCase):
    def test_elmo_lstm(self):
        context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
        inputs = Tensor(np.random.randn(3, 10, 10), mindspore.float32)
        backward = Tensor(np.random.randn(3, 10, 10), mindspore.float32)
        hx = Tensor(np.random.randn(4, 3, 30), mindspore.float32)
        cx = Tensor(np.random.randn(4, 3, 20), mindspore.float32)
        lstm = ELMoLSTM(10, 20, 30, 2, 0.5, 1.0, 1.0, True, True, True)

        outputs, (hy, cy) = lstm(inputs, backward, (hx, cx))

        # (num_layers, batch_size, seq_length, hidden_size)
        assert outputs.shape == (2, 3, 10, 60)
        # (num_layers, batch_size, hidden_size)
        assert hy.shape == (2, 3, 60)
        # (num_layers, batch_size, hidden_size)
        assert cy.shape == (2, 3, 40)

    def test_elmo_lstm_batch_first(self):
        context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
        inputs = Tensor(np.random.randn(3, 10, 10), mindspore.float32)
        backward = Tensor(np.random.randn(3, 10, 10), mindspore.float32)
        hx = Tensor(np.random.randn(4, 3, 30), mindspore.float32)
        cx = Tensor(np.random.randn(4, 3, 20), mindspore.float32)
        lstm = ELMoLSTM(10, 20, 30, 2, 0.5, 1.0, 1.0, True, True, True)

        outputs, (hy, cy) = lstm(inputs, backward, (hx, cx))

        # (num_layers, batch_size, seq_length, hidden_size)
        assert outputs.shape == (2, 3, 10, 60)
        # (num_layers, batch_size, hidden_size)
        assert hy.shape == (2, 3, 60)
        # (num_layers, batch_size, hidden_size)
        assert cy.shape == (2, 3, 40)

    def test_elmo_lstm_train_one_step(self):
        context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
        inputs = Tensor(np.random.randn(3, 10, 10), mindspore.float32)
        backward = Tensor(np.random.randn(3, 10, 10), mindspore.float32)
        hx = Tensor(np.random.randn(4, 3, 30), mindspore.float32)
        cx = Tensor(np.random.randn(4, 3, 20), mindspore.float32)
        lstm = ELMoLSTM(10, 20, 30, 2, 0.5, 1.0, 1.0, True, True, True)

        outputs, (hy, cy) = lstm(inputs, backward, (hx, cx))

        
        # (num_layers, batch_size, seq_length, hidden_size)
        assert outputs.shape == (2, 3, 10, 60)
        # (num_layers, batch_size, hidden_size)
        assert hy.shape == (2, 3, 60)
        # (num_layers, batch_size, hidden_size)
        assert cy.shape == (2, 3, 40)