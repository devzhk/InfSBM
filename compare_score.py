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


def vis_score(score_fn, fig_dir, t, device):

    xs = torch.linspace(3, 5, steps=100)
    ys = torch.linspace(3, 5, steps=100)
    x, y = torch.meshgrid(xs, ys, indexing='ij')
    pts = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1).to(device)

    ts = torch.ones(pts.shape[0], device=device) * t

    scores = score_fn(pts, ts)
    z1 = scores[..., 0].reshape(100, 100).cpu()
    z2 = scores[..., 1].reshape(100, 100).cpu()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x.numpy(), y.numpy(), z1.numpy())
    x_path = os.path.join(fig_dir, f'score-x-{t}.png')
    y_path = os.path.join(fig_dir, f'score-y-{t}.png')
    plt.savefig(x_path)
    plt.close()

    ax = plt.axes(projection='3d')
    ax.plot_surface(x.numpy(), y.numpy(), z2.numpy())
    plt.savefig(y_path)
    plt.close()


def score_error(score_fn, true_fn, t0, fig_dir, device):
    dx = 2 / 100
    xs = torch.linspace(3, 5, steps=100)
    ys = torch.linspace(3, 5, steps=100)
    x, y = torch.meshgrid(xs, ys, indexing='ij')
    pts = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1).to(device)

    N = 1000
    dt = 1 / N
    error = 0
    err_list = []
    tspace = torch.linspace(1.0, t0, N, device=device)
    for t in tspace:
        ts = torch.ones(pts.shape[0], device=device) * t
        score = score_fn(pts, ts)
        true_score = true_fn(pts, ts)
        t_error = torch.square(score - true_score)
        t_error = torch.sum(t_error) * dx * dx * dt
        error += t_error.item()
        err_list.append(error)

    print(f'L2 error: {error}')
    fig_path = os.path.join(fig_dir, 'score_error.png')
    plt.plot(tspace.cpu().numpy(), err_list)
    plt.yscale('log')
    plt.ylabel('Accumulated L2 error')
    plt.xlabel('Time')
    plt.savefig(fig_path)


@torch.no_grad()
def subprocess(args):
    t = args.t
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    config = dict2namespace(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # set up directories
    basedir = config.log.basedir
    basedir = os.path.join('exp', basedir)
    os.makedirs(basedir, exist_ok=True)
    vis_dir = os.path.join(basedir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    # set up learned model
    model = MLP(config).to(device)
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt['model'])
    sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max)

    score_fn = get_score_fn(sde, model, continuous=True)
    # visualize score
    vis_score(score_fn, vis_dir, t=t,
              device=device)
    # set up true score
    mean = torch.tensor(config.data.mean, dtype=torch.float32)
    std = torch.tensor(config.data.std, dtype=torch.float32)

    true_model = Gscore(data_mean=mean, data_var=std * std, device=device)
    true_score_fn = get_score_fn(sde, true_model, continuous=True)

    vis_score(true_score_fn, basedir, t=t, device=device)
    # compute error in score
    
    score_error(score_fn, true_score_fn,t0=t, fig_dir=vis_dir, device=device)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/gaussian-8196-xl.yml')
    parser.add_argument('--ckpt', type=str,
                        default='exp/train-8196/ckpts/model-6000.pt')
    parser.add_argument('--t', type=float, default=0.5)
    args = parser.parse_args()
    subprocess(args)
    # truth(args)
