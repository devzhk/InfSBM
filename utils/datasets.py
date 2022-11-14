import torch
from torch.utils.data import Dataset


class Gaussian(Dataset):
    def __init__(self, num_samples, mean, std) -> None:
        super().__init__()
        self.data = mean[None, :] + std[None, :] * \
            torch.randn((num_samples, mean.shape[0]))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]
