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
        if t.shape == torch.Size([]):
            t = torch.ones((x.shape[0],), device=x.device) * t
        y = torch.cat([x, t[:, None]], dim=-1)
        for i, layer in enumerate(self.layers):
            y = layer(y)
            if i < len(self.layers):
                y = self.act(y)
        return y



class VEPrecond(nn.Module):
    def __init__(self, config, sigma_min=0.02, sigma_max=100):
        super().__init__()
        self.model = MLP(config)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def forward(self, x, sigma, labels=None):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32)
        c_out = sigma
        c_noise = (0.5 * sigma).log()

        F_x = self.model(x, c_noise)
        D_x = x + c_out * F_x
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
