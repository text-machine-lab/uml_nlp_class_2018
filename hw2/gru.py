import torch
import torch.cuda
import torch.nn.functional as F
from torch.nn import Parameter


class GRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ##############################
        ### Insert your code below ###
        # create the weight matrices and biases. Use the `torch.nn.Parameter` class
        ##############################
        raise NotImplementedError()

        ###############################
        ### Insert your code above ####
        ###############################

    def forward(self, inputs, hidden):
        """
        Perform a single timestep of a GRU cell using the provided input and the hidden state
        :param inputs: Current input
        :param hidden: Hidden state from the previous timestep
        :return: New hidden state
        """
        ##############################
        ### Insert your code below ###
        # Perform the calculation according to the reference paper (see the README)
        # hidden_new is the new hidden state at the current timestep
        ##############################
        raise NotImplementedError()

        ###############################
        ### Insert your code above ####
        ###############################
        return hidden_new