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


def score_error(score_fn, true_fn, t0, device):
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

    return err_list, tspace.cpu().numpy()


ckpt_list = ['exp/train-8196-xl/ckpts/model-10000.pt', 
             'exp/train-1024-xl/ckpts/model-10000.pt', 
             'exp/train-256-xl/ckpts/model-10000.pt']


config_path = 'configs/gaussian-8196-xl.yml'

with open(config_path, 'r') as f:
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
sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max)

# set up true score
mean = torch.tensor(config.data.mean, dtype=torch.float32)
std = torch.tensor(config.data.std, dtype=torch.float32)

true_model = Gscore(data_mean=mean, data_var=std * std, device=device)
true_score_fn = get_score_fn(sde, true_model, continuous=True)

for ckpt_path in ckpt_list:
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    score_fn = get_score_fn(sde, model, continuous=True)
    errs, ts = score_error(score_fn, true_score_fn, t0=0.00001, device=device)
    plt.plot(ts, errs, label=ckpt_path)
plt.legend()
plt.yscale('log')
plt.ylabel('Accumulated L2 error')
plt.xlabel('Time')
plt.savefig('exp/compare_error.png')