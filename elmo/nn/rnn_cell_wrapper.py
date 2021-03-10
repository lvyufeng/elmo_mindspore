import mindspore
import mindspore.nn as nn
import mindspore.ops as P

class DropoutWrapper(nn.Cell):
    def __init__(
        self,
        cell,
        input_keep_prob=1.0,
        output_keep_prob=1.0,
        state_keep_prob=1.0
        ):
        super().__init__()
        self._input_dropout = nn.Dropout(input_keep_prob)
        self._output_dropout = nn.Dropout(output_keep_prob)
        self._state_dropout = nn.Dropout(state_keep_prob)
        self.cell = cell
        self.cell_type = cell.cell_type
    def construct(self, inputs, state):
        inputs = self._input_dropout(inputs)
        outputs, new_state = self.cell(inputs, state)
        outputs = self._output_dropout(outputs)
        new_state = self._state_dropout(new_state)

        return outputs, new_state

class ResidualWrapper(nn.Cell):
    def __init__(self, cell):
        super().__init__()
        self.cell = cell
        self.cell_type = cell.cell_type
    def construct(self, inputs, state):
        outputs, new_state = self.cell(inputs, state)
        res_outputs = inputs + outputs
        return res_outputs, new_state
            