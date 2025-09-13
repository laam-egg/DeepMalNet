import torch
from torch import nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)  # predicted probability for the true class
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean() if self.reduction == "mean" else focal.sum()
