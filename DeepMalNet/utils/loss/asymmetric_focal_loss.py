import torch
from torch import nn

class AsymmetricFocalLoss(nn.Module):
    def __init__(self, alpha_pos=0.25, alpha_neg=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha = torch.where(targets == 1, self.alpha_pos, self.alpha_neg)
        focal = alpha * (1 - pt) ** self.gamma * bce
        return focal.mean() if self.reduction == "mean" else focal.sum()
