import torch
import torch.nn as nn

from .utils import get_act


class MLP(nn.Module):
    def __init__(self, config):
        """Simple MLP

        Args:
            config (namespace): namespace of model config
        """
        super(MLP, self).__init__()
        layers = config.model.layers
        self.act = get_act(config.model.act)
        self.layers = nn.ModuleList([
            nn.Linear(in_ch, out_ch) for in_ch, out_ch in zip(layers, layers[1:])
        ])

    def forward(self, x, t):
        y = torch.cat([x, t[:, None]], dim=2)
        for i, layer in enumerate(self.layers):
            y = layer(y)
            if i < len(self.layers):
                y = self.act(y)
        return y
