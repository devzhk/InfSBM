import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns


def plot(ts, samples, xlabel, ylabel, title='', savepath='exp/default/test.png'):
    ts = ts.cpu()
    samples = samples.squeeze().t().cpu()
    plt.figure()
    for i, sample in enumerate(samples):
        plt.plot(ts, sample, label=f'sample {i}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(prop={'size': 5})
    plt.savefig(savepath)


def plot_samples(samples, data_samples, save_path):
    x_array = np.concatenate(
        (samples[..., 0], data_samples[..., 0]), axis=None)
    y_array = np.concatenate(
        (samples[..., 1], data_samples[..., 1]), axis=None)
    dat = {'x': x_array,
           'y': y_array,
           'label': ['generated'] * samples.shape[0] + ['data'] * data_samples.shape[0]}

    df = pd.DataFrame(data=dat)
    fig = sns.jointplot(data=df, x='x', y='y', kind='scatter', hue='label')
    fig.savefig(save_path)
