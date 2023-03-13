#%%
import os
import numpy as np
import torch
from omegaconf import OmegaConf
from models.mlp import MLP
from utils.datasets import Gaussian

import matplotlib.pyplot as plt
#%%

def get_unnorm_pdf(x, sigma):
    '''
    unnormalized PDF of Gaussian
    Args:
        - x: N by d,  N d-dimensional 
        - sigma: noise level
    Return:
        - pdf: N-dimensional array
    '''
    in_prod = np.sum(x ** 2, axis=-1)
    sigma2 = 2 * sigma ** 2
    pdf = np.exp(- in_prod / sigma2)
    return pdf


def idealscore(x, data, sigma):
    '''
    score(x, \sigma) = 1/\sigma^2 (\sum_{i} N(x;))

    Args:
        - x: d dimensional array
        - y: N d-dimensional data samples
        - sigma: noise level
    Return:
        - score: d-dimensional array
    '''
    mean = data - x[None, :] 
    unnorm_pdf = get_unnorm_pdf(mean, sigma)
    numer = np.sum(unnorm_pdf[:, None] * mean, axis=0)   # d-dimensional 
    denom = np.sum(unnorm_pdf) 
    return numer / denom / sigma ** 2


def get_data(num_samples):
    config_path = 'configs/1d/gs.yml'
    config = OmegaConf.load(config_path)
    data_mean = torch.tensor(config.data.mean)
    data_std = torch.tensor(config.data.std)
    dataset = Gaussian(num_samples=num_samples, mean=data_mean, std=data_std)
    data = dataset.data.numpy()
    return data


def get_true_score(x, sigmas):
    '''
    return array like sigmas
    '''
    config_path = 'configs/1d/gs.yml'
    config = OmegaConf.load(config_path)
    data_mean = np.array(config.data.mean)
    data_std = np.array(config.data.std)

    sigma2s = data_std ** 2 + sigmas ** 2
    return (data_mean - x) / sigma2s



@torch.no_grad()
def get_ve_score(model, x, sigmas):
    device = torch.device('cuda')

    sigmas = torch.from_numpy(sigmas).to(device).float()
    c_noise = (0.5 * sigmas).log()
    vals = x * torch.ones_like(sigmas)[:, None]
    scores = model(vals, c_noise) / sigmas[:, None]
    return scores.flatten().cpu().numpy()


def load_model(config='configs/1d/edm-gs.yml'):
    config = OmegaConf.load(config)
    device = torch.device('cuda')
    ckpt_dir = os.path.join('exp', config.log.basedir, 'ckpts')
    ckpt_path = os.path.join(ckpt_dir, 'model-9000.pt')
    ckpt = torch.load(ckpt_path, map_location=device)

    model = MLP(config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model

#%%
def gen_score_plot(val, xlim=0.25, start=0, eps=0.01, scale='linear'):
    x = np.array([val, ])

    eps = 0.0
    sigmas = np.linspace(eps, xlim, num=50)
    true_scores = get_true_score(x, sigmas)

    model = load_model()
    learned_scores = get_ve_score(model, val, sigmas)

    plt.plot(sigmas, true_scores, label='Ground truth')
    plt.plot(sigmas, learned_scores, label='VE-SDE')
    scores = np.zeros_like(sigmas)

    num_samples = [10, 1_000, 10_000, 100_000, 1000_000, 10_000_000]
    for num_sample in num_samples[start:]:
        data = get_data(num_sample)
        for i, sigma in enumerate(sigmas):
            scores[i] = idealscore(x, data, sigma)
        plt.plot(sigmas, scores, label=f'{num_sample} samples')

    plt.legend()
    plt.xlabel('noise level: sigma')
    plt.ylabel('Score')
    plt.yscale(scale)
    plt.title(f'Score at x={x[0]}')
    plt.savefig(f'figs/val{val}-eps{eps}.png')
# %%
gen_score_plot(2.5, xlim=0.1, start=1)
# %%
