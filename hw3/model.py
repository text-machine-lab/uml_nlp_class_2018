import numpy as np
import torch
import torch.nn.functional as F

from utils import get_sequences_lengths, variable, argmax


class Seq2SeqModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, padding_idx, init_idx, max_len, teacher_forcing):
        """
        Sequence-to-sequence model
        :param vocab_size: the size of the vocabulary
        :param embedding_dim: Dimension of the embeddings
        :param hidden_size: The size of the encoder and the decoder
        :param padding_idx: Index of the special pad token
        :param init_idx: Index of the <s> token
        :param max_len: Maximum length of a sentence in tokens
        :param teacher_forcing: Probability of teacher forcing
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.padding_idx = padding_idx
        self.init_idx = init_idx
        self.max_len = max_len
        self.teacher_forcing = teacher_forcing

        ##############################
        ### Insert your code below ###
        ##############################
        raise NotImplementedError()

        ###############################
        ### Insert your code above ####
        ###############################


    def zero_state(self, batch_size):
        """
        Create an initial hidden state of zeros
        :param batch_size: the batch size
        :return: A tuple of two tensors (h and c) of zeros of the shape of (batch_size x hidden_size)
        """

        # The axes semantics are (num_layers, batch_size, hidden_dim)
        nb_layers = 1
        state_shape = (nb_layers, batch_size, self.hidden_size)

        ##############################
        ### Insert your code below ###
        ##############################
        raise NotImplementedError()

        ###############################
        ### Insert your code above ####
        ###############################

        return h0, c0

    def encode_sentence(self, inputs):
        """
        Encode input sentences input a batch of hidden vectors z
        :param inputs: A tensor of size (batch_size x max_len) of indices of input sentences' tokens
        :return: A tensor of size (batch_size x hidden_size)
        """

        batch_size = inputs.size(0)

        ##############################
        ### Insert your code below ###
        ##############################
        raise NotImplementedError()

        ###############################
        ### Insert your code above ####
        ###############################

        return z

    def decoder_state(self, z):
        """
        Create initial hidden state for the decoder based on the hidden vectors z
        :param inputs: A tensor of size (batch_size x hidden_size)
        :return: A tuple of two tensors (h and c) of size (batch_size x hidden_size)
        """

        batch_size = z.size(0)

        state_shape = (batch_size, self.hidden_size)
        ##############################
        ### Insert your code below ###
        ##############################
        raise NotImplementedError()

        ###############################
        ### Insert your code above ####
        ###############################

        return z, c0

    def decoder_initial_inputs(self, batch_size):
        """
        Create initial input the decoder on the first timestep
        :param inputs: The size of the batch
        :return: A vector of size (batch_size, ) filled with the index of self.init_idx
        """

        ##############################
        ### Insert your code below ###
        ##############################
        raise NotImplementedError()

        ###############################
        ### Insert your code above ####
        ###############################
        return inputs

    def decode_sentence(self, z, targets=None):
        """
        Decode the tranlation of the input sentences based in the hidden vectors z and return non-normalized probabilities of the output tokens at each timestep
        :param inputs: A tensor of size (batch_size x hidden_size)
        :return: A tensor of size (batch_size x max_len x vocab_size)
        """

        batch_size = z.size(0)

        ##############################
        ### Insert your code below ###
        ##############################
        raise NotImplementedError()

        ###############################
        ### Insert your code above ####
        ###############################

        return outputs

    def forward(self, inputs, targets=None):
        """
        Perform the forward pass of the network and return non-normalized probabilities of the output tokens at each timestep
        :param inputs: A tensor of size (batch_size x max_len) of indices of input sentences' tokens
        :return: A tensor of size (batch_size x max_len x vocab_size)
        """

        if self.training and np.random.rand() < self.teacher_forcing:
            targets = inputs
        else:
            targets = None

        z = self.encode_sentence(inputs)
        outputs = self.decode_sentence(z, targets)
        return outputs
