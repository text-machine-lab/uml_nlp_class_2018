import os

import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from dataset import TwitterFileArchiveDataset
from gru import GRUCell
from utils import init_weights, argmax, cuda, variable, get_sequence_from_indices


class NeuralLanguageModel(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, init_token, eos_token, teacher_forcing=0.7):
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.teacher_forcing = teacher_forcing
        self.init_token = init_token
        self.eos_token = eos_token

        ##############################
        ### Insert your code below ###
        # create an embeding layer, a GRU cell, and the output projection layer
        ##############################
        raise NotImplementedError('Implement the `__init__` method')

        ###############################
        ### Insert your code above ####
        ###############################

    def cell_zero_state(self, batch_size):
        """
        Create an initial hidden state of zeros
        :param batch_size: the batch size
        :return: A tensor of zeros of the shape of (batch_size x hidden_size)
        """
        weight = next(self.parameters()).data
        hidden = Variable(weight.new(batch_size, self.hidden_size).zero_())
        return hidden

    def forward(self, inputs):
        """
        Perform the forward pass of the network and return non-normalized probabilities of the output tokens at each timestep
        :param inputs: A tensor of size (batch_size x max_len) of indices of tweets' tokens
        :return: A tensor of size (batch_size x max_len x vocab_size)
        """
        batch_size, max_len = inputs.shape

        hidden = self.cell_zero_state(batch_size)
        x_i = variable(np.full((1,), self.init_token)).expand((batch_size,))

        outputs = []
        for i in range(max_len):
            ##############################
            ### Insert your code below ###
            # `output` should be the output of the network at the current timestep
            ##############################
            raise NotImplementedError('Implement the `forward` method')

            ###############################
            ### Insert your code above ####
            ###############################
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)

        return outputs

    def produce(self, start_tokens=None, max_len=20):
        """
        Generate a tweet using the provided start tokens at the inputs on the initial timesteps
        :param start_tokens: A tensor of the shape (n,) where n is the number of start tokens
        :param max_len: Maximum length of the tweet
        :return: Indices of the tokens of the generated tweet
        """
        hidden = self.cell_zero_state(1)
        x_i = variable(np.full((1,), self.init_token))

        if start_tokens is not None:
            start_tokens = variable(start_tokens)

        outputs = []
        for i in range(max_len):
            ##############################
            ### Insert your code below ###
            # `output` should be the output of the network at the current timestep
            ##############################
            raise NotImplementedError('Implement the `produce` method')

            ###############################
            ### Insert your code above ####
            ###############################
            outputs.append(output)

        outputs = torch.cat(outputs)
        return outputs


def main():
    max_len = 20
    embedding_size = 200
    hidden_size = 300
    batch_size = 64
    nb_epochs = 500
    max_grad_norm = 5
    teacher_forcing = 0.7

    # filename = 'data/obama_white_house_tweets.txt'
    filename = 'data/trump_tweets.txt'
    dataset = TwitterFileArchiveDataset(filename, max_len=max_len)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print('Data: {}, vocab: {}'.format(len(dataset), len(dataset.token2id)))

    vocab_size = len(dataset.token2id)
    model = NeuralLanguageModel(
        embedding_size, hidden_size, vocab_size,
        dataset.token2id[dataset.INIT_TOKEN], dataset.token2id[dataset.EOS_TOKEN],
        teacher_forcing
    )
    model = cuda(model)
    init_weights(model)

    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, amsgrad=True)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset.token2id[dataset.PAD_TOKEN])

    for epoch in range(nb_epochs):
        epoch_loss = []
        model.train()
        for i, inputs in enumerate(data_loader):
            optimizer.zero_grad()

            inputs = variable(inputs)

            outputs = model(inputs)

            targets = inputs.view(-1)
            outputs = outputs.view(targets.size(0), -1)

            loss = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm(parameters, max_grad_norm)

            optimizer.step()

            epoch_loss.append(float(loss))

        epoch_loss = np.mean(epoch_loss)
        print('Epoch {} loss {}'.format(epoch, epoch_loss))

        # decode something
        model.eval()

        possible_start_tokens = [
            ['Global', ],
            ['Fake', ],
        ]
        start_tokens = possible_start_tokens[np.random.randint(len(possible_start_tokens))]
        start_tokens = np.array([dataset.token2id[t] for t in start_tokens])
        outputs = model.produce(start_tokens, max_len=20)
        outputs = outputs.cpu().numpy()

        produced_sequence = get_sequence_from_indices(outputs, dataset.id2token)
        print('{}'.format(produced_sequence))

        model.train()


if __name__ == '__main__':
    main()
