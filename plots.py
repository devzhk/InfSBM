import matplotlib.pyplot as plt


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
