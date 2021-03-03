import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
from mindspore.ops import operations as P

def orthogonal(tensor, gain=1):
    r"""
    Description:
    Fills the input `Tensor` with a (semi) orthogonal matrix, as
    described in `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013). The input tensor must have
    at least 2 dimensions, and for tensors with more than 2 dimensions the
    trailing dimensions are flattened.

    Args:
        tensor: an n-dimensional `mindspore.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor

    Examples:
        >>> w = Tensor(np.empty([3, 5]), mindspore.float32)
        >>> orthogonal(w)
    """
   
    size_op = P.Size()
    transpose = P.Transpose()
    mul = P.Mul()

    if len(tensor.shape) < 2:
       raise ValueError("Only tensors with 2 or more dimensions are supported") 

    rows = tensor.shape[0]
    cols = size_op(tensor) // rows
    flatten = Tensor(np.random.normal(size=(rows, cols)), mindspore.float32 )
    if rows < cols:
        flatten = transpose(flatten, (1, 0))
        
    #compute the qr factorization
    q, r = np.linalg.qr(flatten.asnumpy())
    d = np.diag(r, 0)
    
    ph = np.sign(d)
    q *= ph
    q = Tensor(q, mindspore.float32)
    
    if rows < cols:
        q = transpose(q, (1, 0))
    
    q = mul(q, gain)
    return q
    
