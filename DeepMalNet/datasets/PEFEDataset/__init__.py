import torch
from torch.utils.data import Dataset
import lmdb
import gc
import msgpack
import msgpack_numpy
# import numpy as np
# from numpy import ndarray

msgpack_numpy.patch()

class PEFEDataset(Dataset):
    def __init__(self, lmdb_path, split="train"):
        # type: (PEFEDataset, str, str) -> None

        super(PEFEDataset, self).__init__()

        self.lmdb_path = lmdb_path

        self.env = None

        self.keys_file_path = f"{lmdb_path}/splits/{split.lower()}_keys.txt"
        gc.collect()
        with open(self.keys_file_path, 'r') as keys_file:
            self.keys = eval("[" + keys_file.read() + "]") # type: list[bytes]
        gc.collect()
    
    def _init_db(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=65536,
            )

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        self._init_db()

        key = self.keys[idx]
        with self.env.begin(write=False) as txn:
            raw_value = txn.get(key)
        payload = msgpack.unpackb(raw_value, raw=False)

        label = payload['lb']
        features = payload['ef']

        X = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.float32)

        return X, y
    
    def __del__(self):
        if self.env is not None:
            self.env.close()
