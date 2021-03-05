import mindspore
import mindspore.nn as nn
import mindspore.ops as P

class ElmoLSTM(nn.Cell):
    def __init__(self, 
                input_size, 
                hidden_size, 
                cell_size, 
                num_layers, 
                dropout:float=0.0,
                cell_clip:float=0.0,
                proj_clip:float=0.0,
                skip_connections:bool=False):
        super().__init__()

    def construct(self,):
        pass