import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.5, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input:  (B, C, H, W) - raw logits
        target: (B, H, W)    - class indices
        """
        if input.dim() != 4:
            raise ValueError("Expected input of shape (B, C, H, W)")

        B, C, H, W = input.shape
        logpt = F.log_softmax(input, dim=1)      # shape: (B, C, H, W)
        pt = logpt.exp()

        # Flatten spatial dimensions
        logpt = logpt.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C)
        pt = pt.permute(0, 2, 3, 1).reshape(-1, C)        # (N, C)
        target = target.view(-1)                         # (N,)

        # Create valid mask
        valid_mask = (target != self.ignore_index)
        target = target[valid_mask]
        logpt = logpt[valid_mask]
        pt = pt[valid_mask]

        # Gather probabilities and log-probabilities for true classes
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)

        # Apply class weights if provided
        if self.weight is not None:
            weight = self.weight[target]
            loss = -weight * (1 - pt) ** self.gamma * logpt
        else:
            loss = - (1 - pt) ** self.gamma * logpt

        return loss.mean()
