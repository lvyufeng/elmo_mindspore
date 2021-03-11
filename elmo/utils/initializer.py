import numpy as np
from mindspore.common.initializer import initializer, Constant, Normal

def init_dense(dense, dim, bias_constant=0.):
    dense.weight.set_data(initializer(Normal(np.sqrt(1.0 / dim)), dense.weight.shape))
    if dense.has_bias:
        dense.bias.set_data(initializer(Constant(bias_constant), [dense.out_channels]))

def init_conv(conv, ):
    pass