import mindspore
import mindspore.nn as nn
import mindspore.ops as P
from elmo.nn.layers import Dense
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
    Parameters
    ----------
    input_dim : ``int``
        The dimensionality of :math:`x`.  We assume the input has shape ``(batch_size, ...,
        input_dim)``.
    num_layers : ``int``, optional (default=``1``)
        The number of highway layers to apply to the input.
    activation : ``Callable[[torch.Tensor], torch.Tensor]``, optional (default=``torch.nn.functional.relu``)
        The non-linearity to use in the highway layers.
    """
    def __init__(self, input_dim: int, num_layers: int=1, activation:str='relu'):
        super().__init__()
        self._input_dim = input_dim
        self._layers = nn.CellList([Dense(input_dim, input * 2) for _ in range(num_layers)])
        self._activation = activation_map[activation]
        for layer in self._layers:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, so we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            data = layer.bias.data
            data[input_dim:] = 1
            layer.bias.set_data(data)

    def construct(self, inputs):
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part, gate = P.Split(1, 2)(projected_input)
            nonlinear_part = self._activation(nonlinear_part)
            gate = P.Sigmoid()(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input

if __name__ == "__main__":
    # wait for test
    pass