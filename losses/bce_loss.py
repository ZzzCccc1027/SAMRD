import torch
import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self, weight=None, reduction="mean"):
        """
        Args:
            weight: Optional weight for positive class (tensor of shape [1] or [B, 1, H, W])
            reduction: 'mean', 'sum' or 'none'
        """
        super(BCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            inputs: logits tensor, shape [B, 1, H, W]
            targets: binary masks, shape [B, 1, H, W]
        Returns:
            Scalar BCE loss
        """
        return self.criterion(inputs, targets)
