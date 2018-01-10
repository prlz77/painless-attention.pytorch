import os
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np


class MatDataset(Dataset):
    def __init__(self, batch_folder, transform=None):
        batches = os.listdir(batch_folder)
        images = []
        labels = []
        for batch in batches:
            data = loadmat(os.path.join(batch_folder, batch))['affNISTdata']
            images += [data['image'].item().reshape((40, 40, 1, -1)).transpose((3, 0, 1, 2))]
            labels += data['label_int'].item()[0].tolist()

        self.images = np.concatenate(images, axis=0)
        self.labels = np.array(labels)
        self.transform = transform

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, item):

        if self.transform is not None:
            images = self.transform(self.images)
        else:
            images = self.images

        return images, self.labels[item]



if __name__ == '__main__':
    dataset = MatDataset('/home/prlz77/DATASETS/affMNIST/test_batches')
    pass