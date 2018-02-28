import string
import pickle

import numpy as np
import torch
from torch.autograd import Variable

from gru import GRUCell


def init_weights(modules):
    if isinstance(modules, torch.nn.Module):
        modules = modules.modules()

    for m in modules:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal(m.weight.data)
            m.bias.data.zero_()

        if isinstance(m, GRUCell):
            weights = [m.W_ir, m.W_hr, m.W_iz, m.W_hz, m.W_in, m.W_hn, ]
            biases = [m.b_ir, m.b_hr, m.b_iz, m.b_hz, m.b_in, m.b_hn, ]

            init_range = 0.01
            for w in weights:
                w.data.uniform_(-init_range, init_range)
            for b in biases:
                b.data.zero_()


def argmax(inputs, dim=-1):
    values, indices = inputs.max(dim=dim)
    return indices


def cuda(obj):
    if torch.cuda.is_available():
        obj = obj.cuda()

    return obj


def variable(obj, volatile=False):
    if isinstance(obj, (list, tuple)):
        return [variable(o, volatile=volatile) for o in obj]

    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)

    obj = cuda(obj)
    obj = Variable(obj, volatile=volatile)
    return obj


def get_sequence_from_indices(indices, id2token):
    tokens = [id2token[idx] for idx in indices]

    tokens = [
        ' ' + t if i != 0 and not t.startswith("'") and not t.startswith("n'") and t not in string.punctuation else t
        for i, t in enumerate(tokens)
    ]
    sequence = ''.join(tokens)

    return sequence
