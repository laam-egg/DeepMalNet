import torch
import torch.nn as nn
import torch.nn.functional as F
from .hyperparams_loader import ScalingHyperparams

class DeepMalNet_Mode:
    TRAINING = "training"
    INFERENCE = "inference"

def deepmalnet_hidden_layers(input_dim, units, dropout, mode):
    # type: (int, int, float, str) -> list[nn.Module]
    """
    Returns a list of neural network layers
    to be added to the DNN.

    units is the number of neurons.

    The returned list always contains 3 layers:

    (input_dim) -> Fully-connected
        -> (units) -> ReLU
        -> (units) -> Batch Normalization
        -> (units) -> Dropout                   (only during training)
        -> (units)
    """

    print(f"({input_dim}) -> Fully-connected")
    print(f"    -> ({units}) -> ReLU")
    print(f"    -> ({units}) -> Batch Normalization")
    # if mode == DeepMalNet_Mode.TRAINING:
    print(f"    -> ({units}) -> Dropout")
    print(f"    -> ({units})")
    print()

    return [
        nn.Linear(input_dim, units),
        nn.ReLU(),
        nn.BatchNorm1d(units),
        # *(
        #     [nn.Dropout(dropout)] if mode == DeepMalNet_Mode.TRAINING
        #     else []
        # ),
        nn.Dropout(dropout),
    ]

class DeepMalNetNNModule(nn.Module):
    def __init__(self, input_dim, mode, dropout=0.01):
        # type: (DeepMalNetNNModule, int, str, float) -> None

        print(f"Establishing DeepMalNetNNModule ---")
        print(f"    - mode      =   {mode}")
        print(f"    - dropout   =   {dropout}")
        print(f"---")
        super(DeepMalNetNNModule, self).__init__()

        layers = []

        hidden_dim_changes = [4608, 4096, 3584, 3072, 2560, 2048, 1536, 1024, 512, 128]
        previous_dim = input_dim
        for hidden_dim in hidden_dim_changes:
            layers.extend(
                deepmalnet_hidden_layers(previous_dim, hidden_dim, dropout, mode)
            )
            previous_dim = hidden_dim

        # Output
        layers.append(nn.Linear(previous_dim, 1))
        print(f"({previous_dim}) -> Fully-connected -> (1)", end="")

        if mode == DeepMalNet_Mode.INFERENCE:
            layers.append(nn.Sigmoid())
            print(f" -> Sigmoid")
        else:
            print(f" -> (raw score)")
        
        # Pack into Sequential
        self.model = nn.Sequential(*layers)

        # Temporary default
        scaling_hyperparams = ScalingHyperparams.load_default()
        mean = scaling_hyperparams.mean
        std = scaling_hyperparams.std
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)
    
    def load_model_data(self, model_data):
        # type: (DeepMalNetNNModule, dict) -> None
        state_dict = model_data["model_state_dict"]
        try:
            mean = state_dict["mean"]
            std = state_dict["std"]
        except KeyError as e:
            print(f"[WARN] Model doesn't have stored mean/std scaling hyperparams ; using defaults")
            scaling_hyperparams = ScalingHyperparams.load_default()
            mean = scaling_hyperparams.mean
            std = scaling_hyperparams.std
        else:
            print(f"[INFO] Loading model's scaling hyperparams...")

        self.load_state_dict(state_dict)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
    
    def get_scaling_hyperparams(self):
        # type: (DeepMalNetNNModule) -> ScalingHyperparams
        return ScalingHyperparams(
            mean=self.mean,
            std=self.std,
        )
