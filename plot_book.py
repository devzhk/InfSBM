import os
import numpy as np
import torch
from omegaconf import OmegaConf

from utils.datasets import Gaussian
from models.mlp import MLP

from matplotlib import cm
import matplotlib.pyplot as plt



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


def visualize_score(num_samples=100, sigma_lim=0.25):

    model = load_model()
    eps = 0.0
    xs = np.linspace(-1, 3, num=41)
    sigmas = np.linspace(0.0, sigma_lim, num=50)
    
    true_scores = np.zeros((xs.shape[0], sigmas.shape[0]))
    learned_scores = np.zeros_like(true_scores)

    for i, x in enumerate(xs):
        true_scores[i] = get_true_score(x, sigmas)
        learned_scores[i] = get_ve_score(model, x, sigmas)

    # plot 
    x_arr, sigma_arr = np.meshgrid(xs, sigmas, indexing='ij')
    
    ax = plt.axes(projection='3d')
    s1 = ax.plot_surface(x_arr, sigma_arr, true_scores, cmap=cm.Blues)

    s2 = ax.plot_surface(x_arr, sigma_arr, learned_scores, cmap=cm.Reds)
    ax.set_xlabel('x')
    ax.set_ylabel('sigma')
    ax.set_zlabel('score')
    ax.view_init(elev=0., azim=-15)
    plt.savefig('figs/vesde-score1d-compare-15.png')


visualize_score()
    


