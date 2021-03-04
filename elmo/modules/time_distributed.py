"""
A wrapper that unrolls the second (time) dimension of a tensor
into the first (batch) dimension, applies some other ``Module``,
and then rolls the time dimension back up.
"""
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as P
import numpy as np

class TimeDistributed(nn.Cell):
    """
    Given an input shaped like ``(batch_size, time_steps, [rest])`` and a ``Module`` that takes
    inputs like ``(batch_size, [rest])``, ``TimeDistributed`` reshapes the input to be
    ``(batch_size * time_steps, [rest])``, applies the contained ``Module``, then reshapes it back.
    """
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self._module = module
        self.reshape = P.Reshape()
    def construct(self, *inputs):
        reshaped_inputs = []
        for input_tensor in inputs:
            input_size = input_tensor.shape
            if len(input_size) < 2:
                raise RuntimeError("No dimension to distribute: " + str(input_size)) 

            # squash batch_size and timesteps into a single axis
            # (batch_size*time_steps, input_size)  
            squashed_shape = (-1,) + input_size[2:]
            reshaped_inputs.append(self.reshape(input_tensor, squashed_shape))
        reshape_outputs = self._module(*reshaped_inputs)

        # get the output back into the right shape
        # (batch_size, time_steps, [hidden_size])
        new_shape = input_size[:2] + reshape_outputs.shape[1:]
        outputs = self.reshape(reshape_outputs, new_shape)

        return outputs

if __name__ == "__main__":
    # wait for test
    pass