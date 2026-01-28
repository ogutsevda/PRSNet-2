import torch
import random
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, balanced_sampling=False):
        X[X == 3] = 0
        self.X = X
        self.Y = Y
        self.balanced_sampling = balanced_sampling
        if self.balanced_sampling:
            self.n_samples = 10000000
            self.classes = torch.unique(self.Y)
            self.class_indices = {
                cls.item(): torch.where(self.Y == cls)[0] for cls in self.classes
            }
        else:
            self.n_samples = len(self.Y)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if self.balanced_sampling:
            cls = random.choice(self.classes)
            sample_indices = self.class_indices[cls.item()]
            index = random.choice(sample_indices)
        x = self.X[index]
        y = self.Y[index]

        return {"x": x, "y": y}
