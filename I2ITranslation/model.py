import math
import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm
from torch.nn import init
from torchvision import models
from typing import List


def weights_init_normal(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net: nn.Module, init_type='normal'):
    net.apply(weights_init_normal)


class CBR(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel: int,
                 stride: int,
                 pad: int,
                 up=False,
                 norm="in",
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


class ResBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 sn=False):

        super(ResBlock, self).__init__()

        self.cbr0 = CBR(in_ch, out_ch, 3, 1, 1, sn=sn)
        self.cbr1 = CBR(out_ch, out_ch, 3, 1, 1, sn=sn)

    def forward(self, x):
        h = self.cbr0(x)
        h = self.cbr1(h)

        return h + x


class Generator(nn.Module):
    def __init__(self,
                 nc_size: int,
                 layers=8,
                 base=64,
                 sn=False):
        super(Generator, self).__init__()

        self.enc = nn.Sequential(
            CBR(3 + nc_size, base, 7, 1, 3, sn=sn),
            CBR(base, base*2, 4, 2, 1, sn=sn),
            CBR(base*2, base*4, 4, 2, 1, sn=sn),
        )
        self.res = self._make_res(base, sn, layers)
        self.dec = nn.Sequential(
            CBR(base*4, base*2, 3, 1, 1, up=True, sn=sn),
            CBR(base*2, base, 3, 1, 1, up=True, sn=sn)
        )

        if sn:
            self.out = nn.Sequential(
                spectral_norm(nn.Conv2d(base, 3, 7, 1, 3)),
                nn.Tanh()
            )
        else:
            self.out = nn.Sequential(
                nn.Conv2d(base, 3, 7, 1, 3),
                nn.Tanh()
            )

        init_weights(self.enc)
        init_weights(self.res)
        init_weights(self.dec)
        init_weights(self.out)

    @staticmethod
    def _make_res(base: int, sn: bool, layers: int):
        modules = [ResBlock(base*4, base*4, sn=sn) for _ in range(layers)]

        return nn.ModuleList(modules)

    @staticmethod
    def _modify(x, c):
        batch, ch, h, w = x.size()
        _, nc = c.size()

        c = c.view(1, 1, batch, nc).repeat(h, w, 1, 1)
        c = c.permute(2, 3, 0, 1)

        return torch.cat([x, c], dim=1)

    def _res(self, x):
        for layer in self.res:
            x = layer(x)

        return x

    def forward(self, x, c):
        x = self._modify(x, c)

        x = self.enc(x)
        x = self._res(x)
        x = self.dec(x)

        return self.out(x)


class Discriminator(nn.Module):
    def __init__(self,
                 nc_size: int,
                 base=64):

        super(Discriminator, self).__init__()

        self.feature = nn.Sequential(
            CBR(3, base, 4, 2, 1, norm="no"),
            CBR(base, base*2, 4, 2, 1, norm="no"),
            CBR(base*2, base*4, 4, 2, 1, norm="no"),
            CBR(base*4, base*8, 4, 2, 1, norm="no"),
            CBR(base*8, base*16, 4, 2, 1, norm="no"),
            CBR(base*16, base*32, 4, 2, 1, norm="no")
        )

        self.cout = nn.Conv2d(base*32, 1, 1, 1, 0)
        self.cinp = nn.Conv2d(base*32, nc_size, 1, 1, 0)
        self.cma0 = nn.Conv2d(base*64 + nc_size, base*32, 1, 1, 0)
        self.cma1 = nn.Conv2d(base*32, base*1, 1, 1, 0)

        self.relu = nn.LeakyReLU()

        init_weights(self.feature)
        init_weights(self.cout)
        init_weights(self.cinp)
        init_weights(self.cma0)
        init_weights(self.cma1)

    @staticmethod
    def _modify(x_feat, y_feat, c):
        batch, ch, h, w = x_feat.size()
        _, nc = c.size()

        c = c.view(1, 1, batch, nc).repeat(h, w, 1, 1)
        c = c.permute(2, 3, 0, 1)

        return torch.cat([x_feat, y_feat, c], dim=1)

    def forward(self, x, y, c, method):
        x_feat = self.feature(x)

        if method == "adv":
            h = self.cout(x_feat)

        if method == "inp":
            hinp = self.cinp(x_feat)
            h = torch.mean(hinp, dim=(2, 3))

        if method == "mat":
            y_feat = self.feature(y)
            h = self._modify(x_feat, y_feat, c)
            h = self.relu(self.cma0(h))
            h = self.cma1(h)

        return h
