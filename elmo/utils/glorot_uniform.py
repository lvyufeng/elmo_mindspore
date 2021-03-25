import numpy as np
from mindspore import Tensor

def glorot_uniform(shape):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = np.random.uniform(-init_range, init_range, shape).astype(np.float32)
    return Tensor(initial)