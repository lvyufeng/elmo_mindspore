import unittest
import mindspore
import numpy as np
from mindspore import Tensor
from elmo.ops.sampled_softmax_loss import SampledSoftmaxLoss
from mindspore import context

class TestSampledSofrmaxLoss(unittest.TestCase):
    
    def test_char_encoder(self):
        context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')
        mindspore.set_seed(1)
        weights = Tensor(np.random.randint(0, 9, [7, 10]), mindspore.float32)
        biases = Tensor(np.random.randint(0, 9, [7]), mindspore.float32)
        labels = Tensor([0, 1, 2])
        inputs = Tensor(np.random.randint(0, 9, [3, 10]), mindspore.float32)

        loss = SampledSoftmaxLoss(num_sampled=4, num_classes=7, num_true=1, seed=1)
        output = loss(weights, biases, labels, inputs)

        assert output.shape == labels.shape

    def test_char_encoder_graph_mode(self):
        context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')

        mindspore.set_seed(1)
        weights = Tensor(np.random.randint(0, 9, [7, 10]), mindspore.float32)
        biases = Tensor(np.random.randint(0, 9, [7]), mindspore.float32)
        labels = Tensor([0, 1, 2])
        inputs = Tensor(np.random.randint(0, 9, [3, 10]), mindspore.float32)

        loss = SampledSoftmaxLoss(num_sampled=4, num_classes=7, num_true=1, seed=1)
        output = loss(weights, biases, labels, inputs)

        assert output.shape == labels.shape
        