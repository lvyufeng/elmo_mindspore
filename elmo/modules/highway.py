import mindspore
import mindspore.nn as nn
import mindspore.ops as P
import numpy as np
from mindspore.common.initializer import Normal, Constant

activation_map = {
    'tanh': P.Tanh(),
    'relu': P.ReLU()
}

class HighWay(nn.Cell):
    """
    A `Highway layer <https://arxiv.org/abs/1505.00387>`_ does a gated combination of a linear
    transformation and a non-linear transformation of its input.  :math:`y = g * x + (1 - g) *
    f(A(x))`, where :math:`A` is a linear transformation, :math:`f` is an element-wise
    non-linearity, and :math:`g` is an element-wise gate, computed as :math:`sigmoid(B(x))`.
    This module will apply a fixed number of highway layers to its input, returning the final
    result.

    Note: Tensorflow version is `y = (1 - g) * x + g * f(x)`, we use the same equation.
    
    Parameters
    ----------
    input_dim : ``int``
        The dimensionality of :math:`x`.  We assume the input has shape ``(batch_size, ...,
        input_dim)``.
    num_layers : ``int``, optional (default=``1``)
        The number of highway layers to apply to the input.
    activation : ``str``, optional (default=``relu``)
        The non-linearity to use in the highway layers.
    """
    def __init__(self, input_dim: int, num_layers: int=1, activation:str='relu'):
        super().__init__()
        self._input_dim = input_dim
        self._layers = []
        for _ in range(num_layers):
            carry = nn.Dense(input_dim, input_dim, weight_init=Normal(np.sqrt(1.0 / input_dim)), bias_init=Constant(-2.0))
            transform = nn.Dense(input_dim, input_dim, weight_init=Normal(np.sqrt(1.0 / input_dim)))
            self._layers.append((carry, transform))
        
        self._activation = activation_map[activation]

    def construct(self, inputs):
        for layer in self._layers:
            carry_gate = layer[0](inputs)
            carry_gate = P.Sigmoid()(carry_gate)
            transform_gate = layer[1](inputs)
            transform_gate = self._activation(transform_gate)
            
            current_input = carry_gate * transform_gate + (1 - carry_gate) * inputs
        return current_input