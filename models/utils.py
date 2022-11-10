import torch.nn as nn


def get_act(name):
    """Get activation functions from the config file."""

    if name == 'elu':
        return nn.ELU()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif name == 'swish':
        return nn.SiLU()
    elif name == 'softplus':
        return nn.Softplus()
    else:
        raise NotImplementedError('activation function does not exist!')
