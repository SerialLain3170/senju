import torch
import numpy as np
import torch.nn as nn

from typing import List
from torchvision import models
from torch.nn import init
from torch.nn.utils import spectral_norm


# Initialization of model
def weights_init_normal(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net: nn.Module):
    net.apply(weights_init_normal)


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 36):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h = self.slice5(h_relu4)

        return [h_relu1, h_relu2, h_relu3, h_relu4, h]


# Basic components of Generator and Discriminator
class CBR(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel: int,
                 stride: int,
                 pad: int,
                 up=False,
                 norm="bn",
                 activ="lrelu",
                 sn=False):

        super(CBR, self).__init__()

        modules = []
        modules = self._preprocess(modules, up)
        modules = self._conv(modules, in_ch, out_ch, kernel, stride, pad, sn)
        modules = self._norm(modules, norm, out_ch)
        modules = self._activ(modules, activ)

        self.cbr = nn.ModuleList(modules)

    @staticmethod
    def _preprocess(modules: List, up: bool) -> List:
        if up:
            modules.append(nn.Upsample(scale_factor=2, mode="bilinear"))

        return modules

    @staticmethod
    def _conv(modules: List,
              in_ch: int,
              out_ch: int,
              kernel: int,
              stride: int,
              pad: int,
              sn: bool) -> List:
        if sn:
            modules.append(spectral_norm(nn.Conv2d(in_ch, out_ch, kernel, stride, pad)))
        else:
            modules.append(nn.Conv2d(in_ch, out_ch, kernel, stride, pad))

        return modules

    @staticmethod
    def _norm(modules: List,
              norm: str,
              out_ch: int) -> List:

        if norm == "bn":
            modules.append(nn.BatchNorm2d(out_ch))
        elif norm == "in":
            modules.append(nn.InstanceNorm2d(out_ch))

        return modules

    @staticmethod
    def _activ(modules: List, activ: str) -> List:
        if activ == "relu":
            modules.append(nn.ReLU())
        elif activ == "lrelu":
            modules.append(nn.LeakyReLU())

        return modules

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.cbr:
            x = layer(x)

        return x


class SpatialAttention(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int):
        super(SpatialAttention, self).__init__()

        self.c0 = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        self.c1 = nn.Conv2d(out_ch, out_ch, 1, 1, 0)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.c0(x))
        x = self.sigmoid(self.c1(x))

        return x


class ChannelAttention(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 k=16):

        super(ChannelAttention, self).__init__()

        self.l0 = nn.Linear(in_ch, int(in_ch / k))
        self.l1 = nn.Linear(int(in_ch / k), in_ch)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.pool(x).squeeze(3).squeeze(2)
        h = self.relu(self.l0(h))
        h = self.l1(h)
        h = h.view(h.size(0), h.size(1), 1, 1)

        return h.expand_as(x)


class ResidualAttentionModule(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(ResidualAttentionModule, self).__init__()

        self.c0 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c1 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.ca = ChannelAttention(out_ch, out_ch)
        self.sa = SpatialAttention(out_ch, out_ch)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.relu(self.c0(x))
        h = self.c1(h)
        sa = self.sa(h)
        ca = self.ca(h)
        hs = self.sigmoid(sa + ca)

        return h * hs


class Generator(nn.Module):
    def __init__(self, base=64, num_layers=16):
        super(Generator, self).__init__()

        self.res = self._make_res(base, num_layers)
        self.cinit = nn.Sequential(
            nn.Conv2d(3, base, 3, 1, 1),
            nn.ReLU()
        )
        self.cmiddle = nn.Conv2d(base, base, 3, 1, 1)

        self.dec = nn.Sequential(
            CBR(base, base, 3, 1, 1, up=True),
            CBR(base, base, 3, 1, 1, up=True),
            nn.Conv2d(base, 3, 3, 1, 1),
            nn.Tanh()
        )

    @staticmethod
    def _make_res(base: int, num_layers: int):
        modules = [ResidualAttentionModule(base, base) for _ in range(num_layers)]
        modules = nn.ModuleList(modules)

        return modules

    def _res(self, x):
        for layer in self.res:
            x = layer(x)

        return x

    def forward(self, x):
        hinit = self.cinit(x)
        h = self._res(hinit)
        h = self.cmiddle(h)
        h = h + hinit

        return self.dec(h)
