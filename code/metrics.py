from typing import Tuple

import torch
from torch.nn.functional import softmax
import numpy as np


def calc_fb_and_metrics(outputs: torch.Tensor, labels: torch.Tensor, true_fb: torch.Tensor, dt_ms: float) \
        -> Tuple[float, float, float, float, np.ndarray]:
    # outputs: FloatTensor[batch_size, height, witdh] with probability of fb
    # labels: LongTensor[batch_size, height, width] with integer values 0 or 1, where 1 is after fb
    # true_fb: LongTensor[batch_size, width] with indices of first breaks
    # dt_ms: float with time discretization step in ms
    # All tensors must be on the same device

    # The 'outputs' tensor may change during calculations, so use a 'tensor.сlone()' to avoid mistakes

    iou = iou_calculation(outputs.clone(), labels)

    pred_fb = fb_calculation(outputs)

    min_diff_fb, max_diff_fb, mean_diff_fb = diff_fb_calculation(pred_fb, true_fb, dt_ms)

    return (iou.cpu().item(),
            min_diff_fb.cpu().item(),
            max_diff_fb.cpu().item(),
            mean_diff_fb.cpu().item(),
            pred_fb.cpu().numpy())


def iou_calculation(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    max_outputs = torch.mean(torch.max(outputs, dim=0)[0])
    smooth = 1e-6
    outputs = torch.ge(outputs, 0.5 * max_outputs)
    labels = torch.eq(labels, 1)
    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))
    iou = ((intersection + smooth) / (union + smooth)).mean()
    return iou


def fb_calculation(outputs: torch.Tensor) -> torch.Tensor:
    max_outputs = torch.mean(torch.max(outputs, dim=0)[0])
    pred_fb = torch.ge(outputs, 0.5 * max_outputs).int().cpu().numpy()
    # torch.argmax return random arg if several maximum appear, np.argmax return first arg in this case
    pred_fb = np.argmax((pred_fb[:, 1:-1, :] - pred_fb[:, 0:-2, :]), axis=1)
    return torch.from_numpy(pred_fb).float().to(outputs.device)


def diff_fb_calculation(pred_fb: torch.Tensor, true_fb: torch.Tensor, dt_ms: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    diff_fb = torch.abs((true_fb.float() - pred_fb) * dt_ms)
    min_diff_fb = torch.mean(torch.min(diff_fb, dim=1)[0])
    max_diff_fb = torch.mean(torch.max(diff_fb, dim=1)[0])
    mean_diff_fb = torch.sqrt(torch.mean(diff_fb ** 2))
    return min_diff_fb, max_diff_fb, mean_diff_fb
