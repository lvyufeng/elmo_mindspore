from typing import Optional, Tuple, List

import mindspore
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C

class LstmCellWithProjection(nn.Cell):
    r"""
    Description:An LSTM with Recurrent Dropout and a projected and clipped hidden state and
    memory. 
    Args:
    ----------
    input_size :  The dimension of the inputs to the LSTM.
    hidden_size : The dimension of the outputs of the LSTM.
    cell_size : The dimension of the memory cell used for the LSTM.
    go_forward: The direction in which the LSTM is applied to the sequence.
        Forwards by default, or backwards if False.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 cell_size: int,
                 go_forward: bool=True,
                 recurrent_droppout_probability:float,
                 memory_cell_clip_value: Optional[float]=None,
                 state_projection_clip_value:Optional[float]=None)->None:

        super(LstmCellWithProjection, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.go_forward = go_forward
        self.recurrent_droppout_probability = recurrent_droppout_probability
        self.memory_cell_clip_value = memory_cell_clip_value
        self.state_projection_clip_value = state_projection_clip_value

        self.input_linearity = nn.Dense(input_size, 4 * cell_size, has_bias=False)
        self.state_linearity = nn.Dense(input_size, 4 * cell_size, has_bias=True)

        self.state_projection = nn.Dense(cell_size, hidden_size, has_bias=False)

        self.squeeze = P.Squeeze(0)
        self.div = P.Div()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.clamp = C.clip_by_value()
        self.unsqueeze = P.ExpandDims(0)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def get_dropout_mask(droppout_probability, tensor_for_masking):
        """
        Computes and returns an element-wise dropout mask for a given tensor, where
        each element in the mask is dropped out with probability dropout_probability.
        numpy
        Args:
        dropout_probability : Probability of dropping a dimension of the input.
        tensor_for_masking : mindspore.Tensor, required.

        """
        binary_mask = Tensor(np.random.normal(size = tensor_for_masking.shape) > \
                             droppout_probability, mindspore.float32)
        dropout_mask = self.div(binary_mask, (1.0-droppout_probability))
        return dropout_mask

    def construct(self,inputs, 
                 batch_lengths,
                 initial_state):
        """
        Args:
        inputs:
        batch_lengths:
        initial_state:
        """
        
        batch_size = input.shape()[0]
        total_timesteps = input.shape[1]

        output_accumulator = Tensor(np.zeros([batch_size, total_timesteps, \
                                    self.hidden_size]), mindspore.float32)

        if initial_state is None:
            full_batch_previous_memory = Tensor(np.zeros([batch_size, self.cell_size]), mindspore.float32)
            full_batch_previous_state = Tensor(np.zeros([batch_size, self.hidden_size]), mindspore.float32)
        else:
            full_batch_previous_state = self.squeeze(initial_state[0])
            full_batch_previous_memory = self.squeeze(initial_state[1])

        if self.go_forward:
            current_length_index = batch_size - 1
        else:
            current_length_index = 0
        #dropout_mask: numpy array
        if self.recurrent_droppout_probability > 0.0 and self.training:
            dropout_mask = get_dropout_mask(self.recurrent_droppout_probability,
                                            full_batch_previous_state)
        else:
            dropout_mask = None
        
        for timestep in range(total_timesteps):
            # The index depends on which end we start.
            index = timestep if self.go_forward else total_timesteps - timestep - 1

            #finding the index into the batch dimension
            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            else:
                while current_length_index < (len(batch_lengths) - 1) and \
                    batch_lengths[current_length_index + 1] > index:
                    current_length_index += 1

            # get the slices of the batch which we
            # need for the computation at this timestep.
            # shape (batch_size, cell_size)
            previous_memory = Tensor(np.copy(full_batch_previous_memory.asnumpy() \
                            [0: current_length_index + 1]), mindspore.float32)
            # shape (batch_size, cell_size)
            previous_state = Tensor(np.copy(full_batch_previous_state.asnumpy() \
                            [0: current_length_index + 1]), mindspore.float32)
            # shape (batch_size, cell_size)
            timestep_input = Tensor(np.copy(inputs.asnumpy() \
                            [0: current_length_index + 1, index]), mindspore.float32)

            # Do the projections for all the gates all at once.
            # Both have shape (batch_size, 4 * cell_size)
            projected_input = self.input_linearity(timestep_input)
            projected_state = self.state_linearity(previous_state)

            # Main LSTM equations using relevant chunks of the big linear
            # projections of the hidden state and inputs
            input_gate = self.sigmoid(projected_input[:, (0 * self.cell_size):(1 * self.cell_size)] +
                                     projected_state[:, (0 * self.cell_size):(1 * self.cell_size)])
            forget_gate = self.sigmoid(projected_input[:, (1 * self.cell_size):(2 * self.cell_size)] +
                                     projected_state[:, (1 * self.cell_size):(2 * self.cell_size)])
            memory_init = self.tanh(projected_input[:, (2 * self.cell_size):(3 * self.cell_size)] +
                                     projected_state[:, (2 * self.cell_size):(3 * self.cell_size)])
            output_gate = self.sigmoid((projected_input[:, (3 * self.cell_size):(4 * self.cell_size)] +
                                     projected_state[:, (3 * self.cell_size):(4 * self.cell_size)])
            memory = input_gate * memory_init + forget_gate * previous_memory

            # Here is the non-standard part of this LSTM cell; first, we clip the
            # memory cell, then we project the output of the timestep to a smaller size
            # and again clip it.
            if self.memory_cell_clip_value:
                memory = self.clamp(memory, 
                        -self.memory_cell_clip_value,
                        self.memory_cell_clip_value)

            # shape (current_length_index, cell_size)
            pre_projection_timestep_output = output_gate * torch.tanh(memory)

            # shape (current_length_index, hidden_size)
            timestep_output = self.state_projection(pre_projection_timestep_output)

            if self.state_projection_clip_value:
                timestep_output  = self.clamp(timestep_output,
                                            -self.state_projection_clip_value,
                                            self.state_projection_clip_value)

            #Only do dropout if the dropout prob is > 0.0 and we are in training mode.
            if dropout_mask is not None:
                timestep_output = timestep_output * dropout_mask[0: current_length_index + 1]

            # create a new variable for the the whole batch at this timestep 
            # and insert the result for the relevant elements of the batch into it.
            # clone
            #full_batch_previous_memory = full_batch_previous_memory
            full_batch_previous_memory[0:current_length_index + 1] = memory
            full_batch_previous_state[0:current_length_index + 1] = timestep_output
            output_accumulator[0:current_length_index + 1, index] = timestep_output

            # shape:(num_layers * num_directions, batch_size, ...). As this
            # LSTM cell cannot be stacked, the first dimension here is just 1.
        
            final_state = (self.unsqueeze(full_batch_previous_state,
                            self.unsqueeze(full_batch_previous_memory)))
            return output_accumulator, final_state

