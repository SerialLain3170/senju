import torch
import torch.nn as nn

from typing import List

maeloss = nn.L1Loss()
mseloss = nn.MSELoss()
softplus = nn.Softplus()


class RAMCalculator:
    def __init__(self):
        pass

    @staticmethod
    def content_loss(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(y - t))

    @staticmethod
    def perceptual_loss(vgg: nn.Module,
                        y: torch.Tensor,
                        t: torch.Tensor) -> torch.Tensor:
        sum_loss = 0
        y_feat = vgg(y)
        t_feat = vgg(t)

        for y, t in zip(y_feat, t_feat):
            _, c, h, w = y.size()
            sum_loss += torch.mean(torch.abs(y-t)) / (c * h * w)

        return sum_loss
