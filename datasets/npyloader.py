import os
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np


class NpyDataset(Dataset):
    def __init__(self, path, transform=None):
        data = np.load(path).item()
        self.images = torch.from_numpy(data['images'])
        self.labels = torch.from_numpy(data['labels'])
        self.transform = transform

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, item):
        return self.images[item].float() / 255, self.labels[item]



if __name__ == '__main__':
    dataset = NpyDataset('/home/prlz77/DATASETS/cluttered-mnist/test.npy')
    pass