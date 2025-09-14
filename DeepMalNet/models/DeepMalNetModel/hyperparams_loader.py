from . import hyperparams
from threading import Lock

class ScalingHyperparams:
    def __init__(self, mean, std):
        # type: (float, float) -> None
        self.mean = mean
        self.std = std
    
    def __repr__(self):
        return f"ScalingHyperparams(mean={self.mean}, std={self.std})"
    
    def __str__(self):
        return self.__repr__()
    
    _DEFAULT = None
    _DEFAULT_LOCK = Lock()

    @classmethod
    def load_default(CLS):
        with CLS._DEFAULT_LOCK:
            if CLS._DEFAULT is None:
                CLS._DEFAULT = CLS._real_load_default()
            return CLS._DEFAULT
    
    @classmethod
    def _real_load_default(CLS):
        import torch
        import numpy as np
        import importlib.resources as pkg_resources

        with pkg_resources.path(hyperparams, "scaling.npz") as hyperparams_path:
            data = np.load(hyperparams_path)
            mean = torch.from_numpy(data["mean"]).float()
            std = torch.from_numpy(data["std"]).float()

        return CLS(mean=mean, std=std)
