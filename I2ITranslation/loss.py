import torch
import torch.nn as nn

from torch import autograd
from utils import call_zeros, call_ones

maeloss = nn.L1Loss()
mseloss = nn.MSELoss()
softplus = nn.Softplus()


class RelGANLossCalculator:
    def __init__(self):
        pass

    @staticmethod
    def _grad_penalty(t_dis: torch.Tensor,
                      t: torch.Tensor) -> torch.Tensor:

        (grad_real,) = autograd.grad(
            outputs=t_dis.sum(), inputs=t, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    def zero_grad_penalty(self,
                          discriminator,
                          y,
                          t):
        y_dis = discriminator(y, y=None, c=None, method="adv")
        t_dis = discriminator(t, y=None, c=None, method="adv")

        loss = self._grad_penalty(y_dis, y)
        loss += self._grad_penalty(t_dis, t)

        return loss

    @staticmethod
    def adversarial_loss_dis(discriminator,
                             y,
                             t):
        y_dis = discriminator(y, y=None, c=None, method="adv")
        t_dis = discriminator(t, y=None, c=None, method="adv")

        loss = nn.ReLU()(1.0 + y_dis).mean() + nn.ReLU()(1.0 - t_dis).mean()

        return loss

    @staticmethod
    def adversarial_loss_gen(discriminator,
                             y):
        y_dis = discriminator(y, y=None, c=None, method="adv")

        return -y_dis.mean()

    @staticmethod
    def interpolation_loss_dis(discriminator,
                               y0,
                               y1,
                               yalpha,
                               alpha,
                               flag):
        if flag:
            y1_dis = discriminator(y1, y=None, c=None, method="inp")
            yalpha_dis = discriminator(yalpha, y=None, c=None, method="inp")

            zeros = call_zeros(y1_dis)

            loss = mseloss(y1_dis, zeros) + mseloss(yalpha_dis, alpha)

        else:
            y0_dis = discriminator(y0, y=None, c=None, method="inp")
            yalpha_dis = discriminator(yalpha, y=None, c=None, method="inp")

            zeros = call_zeros(y0_dis)

            loss = mseloss(y0_dis, zeros) + mseloss(yalpha_dis, alpha)

        return loss

    @staticmethod
    def interpolation_loss_gen(discriminator,
                               yalpha):
        yalpha_dis = discriminator(yalpha, y=None, c=None, method="inp")
        zeros = call_zeros(yalpha_dis)

        return mseloss(yalpha_dis, zeros)

    @staticmethod
    def matching_loss_dis(discriminator,
                          x,
                          y,
                          t,
                          z,
                          v1,
                          v2,
                          v3):
        sr = discriminator(x, t, v1, method="mat")
        sf = discriminator(x, y, v1, method="mat")
        sw0 = discriminator(z, t, v1, method="mat")
        sw1 = discriminator(x, t, v2, method="mat")
        sw2 = discriminator(x, t, v3, method="mat")
        sw3 = discriminator(x, z, v1, method="mat")

        zeros = call_zeros(sr)
        ones = call_ones(sr)

        loss = mseloss(sr, ones)
        loss += mseloss(sf, zeros)
        loss += mseloss(sw0, zeros)
        loss += mseloss(sw1, zeros)
        loss += mseloss(sw2, zeros)
        loss += mseloss(sw3, zeros)

        return loss

    @staticmethod
    def matching_loss_gen(discriminator,
                          x,
                          t,
                          v1):
        sf = discriminator(x, t, v1, method="mat")
        ones = call_ones(sf)

        return mseloss(sf, ones)

    @staticmethod
    def content_loss(y, t):
        return torch.mean(torch.abs(y - t))
