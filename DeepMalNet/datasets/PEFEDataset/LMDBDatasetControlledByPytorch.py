import torch
from torch.utils.data import Dataset
from pefe_ief.dataset import PEFELMDBDataset

class LMDBDatasetControlledByPytorch(Dataset):
    def __init__(self, lmdb_path):
        features, labels = PEFELMDBDataset().read(lmdb_path)
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
