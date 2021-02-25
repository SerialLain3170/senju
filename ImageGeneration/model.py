import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm


class LBR(nn.Module):
    def __init__(self,
                 z_dim: int,
                 out_ch: int,
                 sn=False):
        super(LBR, self).__init__()

        if sn:
            self.l0 = spectral_norm(nn.Linear(z_dim, out_ch))
        else:
            self.l0 = nn.Linear(z_dim, out_ch)

        self.bn0 = nn.BatchNorm2d(out_ch)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.bn0(self.l0(x)))


class ResBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 sn=False):
        super(ResBlock, self).__init__()

        if sn:
            self.c0 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
            self.c1 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        else:
            self.c0 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
            self.c1 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

        self.bn0 = nn.BatchNorm2d(out_ch)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.bn0(self.c0(x)))
        h = self.relu(self.bn1(self.c1(x)))

        return h + x


class UpResBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 sn=False):
        super(UpResBlock, self).__init__()

        if sn:
            self.c0 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
            self.c1 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
            self.c_sc = spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))
        else:
            self.c0 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
            self.c1 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

        self.bn0 = nn.BatchNorm2d(out_ch)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn_sc = nn.BatchNorm2d(out_ch)

        self.relu = nn.LeakyReLU()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")

    def _shortcut(self, x):
        x = self.up(x)
        x = self.relu(self.bn_sc(self.c_sc(x)))

        return x

    def forward(self, x):
        x = self.up(x)
        x = self.relu(self.bn0(self.c0(x)))
        x = self.relu(self.bn1(self.c1(x)))

        return x + self._shortcut(x)


class DownResBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 sn=False):
        super(DownResBlock, self).__init__()

        if sn:
            self.c0 = spectral_norm(nn.Conv2d(in_ch, out_ch, 4, 2, 1))
            self.c1 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
            self.c_sc = spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))
        else:
            self.c0 = nn.Conv2d(in_ch, out_ch, 4, 2, 1)
            self.c1 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

        self.bn0 = nn.BatchNorm2d(out_ch)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn_sc = nn.BatchNorm2d(out_ch)

        self.relu = nn.LeakyReLU()
        self.down = nn.AvgPool2d(2, 2)

    def _shortcut(self, x):
        x = self.down(x)
        x = self.relu(self.bn_sc(self.c_sc(x)))

        return x

    def forward(self, x):
        x = self.relu(self.bn0(self.c0(x)))
        x = self.relu(self.bn1(self.c1(x)))

        return x + self._shortcut(x)


class Generator(nn.Module):
    def __init__(self,
                 z_dim: int,
                 base=64,
                 sn=False):
        super(Generator, self).__init__()

        self.lbr = LBR(z_dim, base*16*4*4, sn=sn)
        self.gen = nn.Sequential(
            UpResBlock(base*16, base*16),
            UpResBlock(base*16, base*8),
            UpResBlock(base*8, base*4),
            UpResBlock(base*4, base*2),
            UpResBlock(base*2, base),
        )

        if sn:
            self.out = nn.Sequential(
                spectral_norm(nn.Conv2d(base, 3, 3, 1, 1)),
                nn.Tanh()
            )
        else:
            self.out = nn.Sequential(
                nn.Conv2d(base, 3, 3, 1, 1),
                nn.Tanh()
            )

    def forward(self, z):
        x = self.relu(self.bn0(self.l0(z)))
        x = self.gen(x)
        return self.out(x)


class Discriminator(nn.Module):
    def __init__(self,
                 base=64,
                 sn=False):
        super(Discriminator, self).__init__()

        self.dis = nn.Sequential(
            DownResBlock(3, base),
            DownResBlock(base, base*2),
            DownResBlock(base*2, base*4),
            DownResBlock(base*4, base*8),
            DownResBlock(base*8, base*16),
        )

        if sn:
            self.out = spectral_norm(nn.Conv2d(base*16, 1, 2, 1, 0))
        else:
            self.out = nn.Conv2d(base*16, 1, 2, 1, 0)

    def forward(self, x):
        return self.out(self.dis(x))
