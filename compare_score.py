import os
import yaml
from argparse import ArgumentParser
import torch
import numpy as np
from models.mlp import MLP
from models.utils import get_score_fn

from models.scores import Gscore
from utils.sde_lib import VPSDE
from utils.helper import dict2namespace, save_ckpt

import matplotlib.pyplot as plt


@torch.no_grad()
def subprocess(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    config = dict2namespace(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MLP(config).to(device)
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt['model'])

    sde = VPSDE()
    score_fn = get_score_fn(sde, model)

    xs = torch.linspace(3, 5, steps=100)
    ys = torch.linspace(3, 5, steps=100)
    x, y = torch.meshgrid(xs, ys, indexing='ij')
    pts = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)
    print(pts.shape)
    ts = torch.zeros(pts.shape[0])

    scores = score_fn(pts, ts)
    z1 = scores[..., 0].reshape(100, 100)
    z2 = scores[..., 1].reshape(100, 100)
    ax = plt.axes(projection='3d')
    ax.plot_surface(x.numpy(), y.numpy(), z1.numpy())
    plt.savefig('exp/score-model1.png')
    ax = plt.axes(projection='3d')
    ax.plot_surface(x.numpy(), y.numpy(), z2.numpy())
    plt.savefig('exp/score-model2.png')


def truth(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    config = dict2namespace(config)
    mean = np.array(config.data.mean)
    std = np.array(config.data.std)
    var = std * std
    gs = Gscore(data_mean=mean, data_var=var)

    xs = torch.linspace(3, 5, steps=100)
    ys = torch.linspace(3, 5, steps=100)
    x, y = torch.meshgrid(xs, ys, indexing='ij')
    pts = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)

    scores = gs(pts.numpy(), 0.0)

    z1 = scores[..., 0].reshape(100, 100)
    z2 = scores[..., 1].reshape(100, 100)
    ax = plt.axes(projection='3d')
    ax.plot_surface(x.numpy(), y.numpy(), z1)
    plt.savefig('exp/true-score-model-x.png')
    ax = plt.axes(projection='3d')
    ax.plot_surface(x.numpy(), y.numpy(), z2)
    plt.savefig('exp/true-score-model-y.png')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/gaussian.yml')
    parser.add_argument('--ckpt', type=str,
                        default='exp/train-256/ckpts/model-5000.pt')
    args = parser.parse_args()
    # subprocess(args)
    truth(args)
