from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# based on:
# https://github.com/arraiyopensource/kornia/blob/master/kornia/losses/focal.py

class FocalLoss(nn.Module):
    alpha: torch.Tensor
    gamma: torch.Tensor
    reduction: Optional[str]

    def __init__(self, alpha: torch.Tensor, gamma: float,
                 reduction: Optional[str] = 'mean') -> None:
        super(FocalLoss, self).__init__()
        self.alpha: torch.Tensor = alpha
        self.gamma: torch.Tensor = torch.tensor(gamma)
        self.reduction: Optional[str] = reduction

    def forward(self, input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input_tensor):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input_tensor)))
        if not len(input_tensor.shape) == 4:
            raise ValueError("Invalid input_tensor shape, we expect BxNxHxW. Got: {}"
                             .format(input_tensor.shape))
        if not input_tensor.shape[-2:] == target.shape[-2:]:
            raise ValueError("input_tensor and target shapes must be the same. Got: {}"
                             .format(input_tensor.shape, input_tensor.shape))
        if not input_tensor.device == target.device:
            raise ValueError(
                "input_tensor and target must be in the same device. Got: {}" .format(
                    input_tensor.device, target.device))

        gamma = self.gamma.to(device=input_tensor.device, dtype=input_tensor.dtype)
        alpha = self.alpha.to(device=input_tensor.device, dtype=input_tensor.dtype).\
            unsqueeze(0).unsqueeze(2).unsqueeze(3).to(dtype=torch.float)

        # compute softmax over the classes axis

        input_soft = F.softmax(input_tensor, dim=1)

        # create the labels one hot tensor
        num_classes = input_tensor.shape[1]
        batch_size, height, width = target.shape
        target_one_hot = torch.zeros(batch_size, num_classes, height, width,
                                     device=input_tensor.device, dtype=input_tensor.dtype)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)

        # compute the actual focal loss
        weight = torch.pow(torch.ones(input_soft.size(), device=input_tensor.device) - input_soft, gamma)

        focal = alpha * weight * torch.log(input_soft)
        loss_tmp = target_one_hot * focal
        loss = F.nll_loss(loss_tmp, target.to(dtype=torch.long).squeeze(1), reduction=self.reduction)

        return loss

