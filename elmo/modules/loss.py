import mindspore
import mindspore.nn as nn
import mindspore.ops as P

class LossCell(nn.Cell):
    def __init__(self, hidden_size, vocab_size, is_training, sample_softmax):
        super().__init__()
        self.is_training = is_training
        self.sample_to_softmax = sample_softmax

        self.proj = nn.Dense(hidden_size, vocab_size)
        