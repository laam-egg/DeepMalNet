from torch.utils.data import DataLoader, random_split
from .LMDBDatasetControlledByPytorch import LMDBDatasetControlledByPytorch
from ...config import config

class PEFEDataset:
    def __init__(self):
        self.lmdb_path = config["ember2024_lmdb_path"]
        self.dataset = LMDBDatasetControlledByPytorch(self.lmdb_path)

        self.train_size = int(0.8 * len(self.dataset))
        self.cv_size = len(self.dataset) - self.train_size
        self.train_dataset, self.cv_dataset = random_split(
            self.dataset,
            [self.train_size, self.cv_size],
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True)
        self.cv_loader = DataLoader(self.cv_dataset, batch_size=256, shuffle=False)
