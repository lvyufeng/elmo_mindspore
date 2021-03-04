import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor, Parameter, ParameterTuple


class Scalar_mix(nn.Cell):
    """
    Computes a paramterised scalar mixture of N tensor, ```mixture = gamma * sum(s_k * tensor_k)```
    where ``s = softmax(w)``, with ``w`` and ``gamma`` scalar parameters.
    """

    def __init__(self, mixture_size: int, do_layer_norm: bool = False) -> None:
        super(Scalar_mix, self).__init__()
        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm
        self.scalar_parameters = ParameterTuple([Parameter(Tensor(np.array([0.0]), mindspore.float32)) \
                                                 for _ in range(mixture_size)])
        self.gamma = Parameter(Tensor(np.array([0.0]), mindspore.float32))
        self.sum = P.ReduceSum()
        self.sqrt = P.Sqrt()
        self.cat = P.Concat()
        self.unsqueeze = P.ExpandDims(0)

    def construct(self, tensors, mask):
        """
        Compute a weighted average of the ``tensors``
        Args:
        tensors: The input tensors can be any shape
        with at least two dimensions, but must all be the same shape.
        mask: When ``do_layer_norm=True``, the ``mask`` is required input.
        for example with ``tensors`` of shape``(batch_size, timesteps, dim)``
        and ``mask`` of shape ``(batch_size, timesteps)``.dtype=mindspore.float32
        """
        if len(tensors) != self.mixture_size:
            raise ValueError("{} tensors were passed, but the module was initialized to "
                             "mix {} tensors.".format(len(tensors), self.mixture_size))

        def _do_layer_norm(tensor, broadcast_mask, num_elments_not_masked):
            tensor_masked = tensor * broadcast_mask
            mean = self.sum(tensor_masked) / num_elments_not_masked
            variance = self.sum(((tensor_masked - mean) * broadcast_mask) ** 2) /
                                num_elments_not_masked
            return (tensor - mean) / self.sqrt(variance + 1E-12)

        normed_weights = P.Softmax(dim=0)(self.cat([parameter for parameter \
                                                    in self.scalar_parameters]))
        normed_weights = P.Split(output_num=normed_weights.shape[0])(normed_weights)  # 待验证 torch.split(split=1)

        if not self.do_layer_norm:
            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * tensor)
            return self.gamma * sum(pieces)
        else:
            # mask_float = mask.float()
            broadcast_mask = self.unsqueeze(mask)
            input_dim = tensors[0].shape[-1]
            num_elments_not_masked = sum(mask) * input_dim

            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * _do_layer_norm(tensor,
                                                      broadcast_mask, num_elments_not_masked))
            return self.gamma * sum(pieces)

if __name__ == "__main__":
    # wait for test
    pass