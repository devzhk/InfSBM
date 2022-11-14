'''
True score of toy problems
'''
import numpy as np
import torch


class Gscore(object):
    def __init__(self, data_mean, data_var,
                 beta_min=0.1, beta_max=20) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.mean = data_mean[None, :]
        self.var = data_var

    def __call__(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * \
            (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        ht = np.exp(log_mean_coeff)
        print(ht)
        print(self.mean.shape)
        print(self.var.shape)
        vec = ht * self.mean - x
        var = np.diag(ht * ht * self.var + (1 - ht * ht))
        inv = np.linalg.inv(var)
        return vec @ inv
