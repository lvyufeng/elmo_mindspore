import unittest
import mindspore
import numpy as np
from mindspore import Tensor
from elmo.modules.highway import HighWay
from mindspore import context

class TestHighWay(unittest.TestCase):
    def test_highway(self):
        context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')
        inputs = Tensor(np.random.randn(3, 10), mindspore.float32)
        highway = HighWay(10, 2)

        outputs = highway(inputs)

        assert outputs.shape == (3, 10)
         
    def test_highway_graph_mode(self):
        context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
        inputs = Tensor(np.random.randn(3, 10), mindspore.float32)
        highway = HighWay(10, 2)

        outputs = highway(inputs)

        assert outputs.shape == (3, 10)