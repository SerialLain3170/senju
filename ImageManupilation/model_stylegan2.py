import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class Vgg19(nn.Module):
    def __init__(self,
                 layer=None,
                 requires_grad=False):

        super(Vgg19, self).__init__()
        self.layer = layer

        vgg_pretrained_features = models.vgg19(pretrained=True).features

        if layer == 'four':
            self.slice = nn.Sequential()
            for x in range(21):
                self.slice.add_module(str(x), vgg_pretrained_features[x])

        elif layer == 'five':
            self.slice = nn.Sequential()
            for x in range(30):
                self.slice.add_module(str(x), vgg_pretrained_features[x])

        else:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer == 'four':
            h = self.slice(x)

        elif self.layer == 'five':
            h = self.slice(x)

        else:
            h_relu1 = self.slice1(x)
            h_relu2 = self.slice2(h_relu1)
            h_relu3 = self.slice3(h_relu2)
            h_relu4 = self.slice4(h_relu3)
            h_relu5 = self.slice5(h_relu4)
            h = h_relu5

        return h


class NoiseInjection(nn.Module):
    def __init__(self):
        super(NoiseInjection, self).__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self,
                x: torch.Tensor,
                noise=None) -> torch.Tensor:
        if noise is None:
            batch, _, height, width = x.size()
            noise = x.new_empty(batch,
                                1,
                                height,
                                width).normal_()

        return x + self.weight * noise


class EqualizedConv2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride=1,
        pad=0,
        bias=True
    ):
        super(EqualizedConv2d, self).__init__()

        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch, kernel, kernel)
        )
        self.scale = 1 / math.sqrt(in_ch * kernel ** 2)

        self.stride = stride
        self.pad = pad

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))

        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.conv2d(
            x,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.pad,
        )

        return out


class EqualizedLinear(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        bias=True,
        bias_init=0,
        lr_mul=1,
    ):
        super(EqualizedLinear, self).__init__()

        self.weight = nn.Parameter(torch.randn(out_ch, in_ch).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_ch).fill_(bias_init))

        else:
            self.bias = None

        self.scale = (1 / math.sqrt(in_ch)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        style_dim: int,
        demodulate=True,
        blur_kernel=[1, 3, 3, 1],
    ):
        super(ModulatedConv2d, self).__init__()

        self.eps = 1e-8
        self.kernel = kernel
        self.in_ch = in_ch
        self.out_ch = out_ch

        fan_in = in_ch * kernel ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.pad = kernel // 2

        self.weight = nn.Parameter(
            torch.randn(1,
                        out_ch,
                        in_ch,
                        kernel,
                        kernel)
        )

        self.modulation = EqualizedLinear(style_dim, in_ch, bias_init=1)
        self.demodulate = demodulate

    def forward(self,
                x: torch.Tensor,
                style: torch.Tensor) -> torch.Tensor:
        batch, in_channel, height, width = x.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_ch, 1, 1, 1)

        weight = weight.view(
            batch * self.out_ch, in_channel, self.kernel, self.kernel
        )

        x = x.view(1, batch * in_channel, height, width)
        out = F.conv2d(x, weight, padding=self.pad, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_ch, height, width)

        return out


class CBR(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel: int,
                 stride: int,
                 pad: int,
                 padding="zero",
                 up=False,
                 norm="in",
                 activ="lrelu",
                 sn=False):

        super(CBR, self).__init__()

        modules = []
        modules = self._preprocess(modules, up)
        modules = self._pad(modules, padding)
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
    def _pad(modules: List, padding: str) -> List:
        if padding == "reflect":
            modules.append(nn.ReflectionPad2d(1))

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
            modules.append(spectral_norm(EqualizedConv2d(in_ch, out_ch, kernel, stride, pad)))
        else:
            modules.append(EqualizedConv2d(in_ch, out_ch, kernel, stride, pad))

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


class StyleCBR(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 texture_dim: int):
        super(StyleCBR, self).__init__()

        self.c0 = ModulatedConv2d(in_ch, out_ch, 3, texture_dim)
        self.injection = NoiseInjection()
        self.relu = nn.LeakyReLU()

    def forward(self,
                x: torch.Tensor,
                s: torch.Tensor) -> torch.Tensor:
        h = self.c0(x, s)
        h = self.injection(h)

        return self.relu(h)


class ResBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 padding="zero",
                 activ="lrelu",
                 down=False,
                 sn=False):
        super(ResBlock, self).__init__()

        modules = []
        modules = self._pad(modules, padding)
        if down:
            modules = self._down(modules, in_ch, out_ch, sn, padding)
        else:
            modules = self._conv(modules, in_ch, out_ch, sn, padding)
        modules = self._activ(modules, activ)
        modules = self._pad(modules, padding)
        modules = self._conv(modules, out_ch, out_ch, sn, padding)
        modules = self._activ(modules, activ)

        self.res = nn.ModuleList(modules)
        self.down = down
        self.downsample = nn.AvgPool2d(3, 2, 1)

        if sn:
            self.c_sc = spectral_norm(EqualizedConv2d(in_ch, out_ch, 1, 1, 0, bias=False))
        else:
            self.c_sc = EqualizedConv2d(in_ch, out_ch, 1, 1, 0, bias=False)

    @staticmethod
    def _pad(modules, padding: str):
        if padding == "reflect":
            modules.append(nn.ReflectionPad2d(1))

        return modules

    @staticmethod
    def _down(modules: List,
              in_ch: int,
              out_ch: int,
              sn: bool,
              padding: str):
        if padding == "reflect":
            if sn:
                modules.append(spectral_norm(EqualizedConv2d(in_ch, out_ch, 4, 2, 0)))
            else:
                modules.append(EqualizedConv2d(in_ch, out_ch, 4, 2, 0))

        else:
            if sn:
                modules.append(spectral_norm(EqualizedConv2d(in_ch, out_ch, 4, 2, 1)))
            else:
                modules.append(EqualizedConv2d(in_ch, out_ch, 4, 2, 1))

        return modules

    @staticmethod
    def _conv(modules: List,
              in_ch: int,
              out_ch: int,
              sn: bool,
              padding: str):
        if padding == "reflect":
            if sn:
                modules.append(spectral_norm(EqualizedConv2d(in_ch, out_ch, 3, 1, 0)))
            else:
                modules.append(EqualizedConv2d(in_ch, out_ch, 3, 1, 0))

        else:
            if sn:
                modules.append(spectral_norm(EqualizedConv2d(in_ch, out_ch, 3, 1, 1)))
            else:
                modules.append(EqualizedConv2d(in_ch, out_ch, 3, 1, 1))

        return modules

    @staticmethod
    def _activ(modules: List, activ: str) -> List:
        if activ == "relu":
            modules.append(nn.ReLU())
        elif activ == "lrelu":
            modules.append(nn.LeakyReLU())

        return modules

    def _skip(self, x: torch.Tensor) -> torch.Tensor:
        if self.down:
            x = self.downsample(x)

        return self.c_sc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self._skip(x)

        for layer in self.res:
            x = layer(x)

        return (x + skip) / math.sqrt(2)


class StyleResBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 texture_dim: int,
                 up=False):

        super(StyleResBlock, self).__init__()

        self.c0 = StyleCBR(in_ch, out_ch, texture_dim)
        self.c1 = StyleCBR(out_ch, out_ch, texture_dim)
        self.c_sc = EqualizedConv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.up = up

    def _skip(self, x: torch.Tensor) -> torch.Tensor:
        if self.up:
            x = self.upsample(x)

        return self.c_sc(x)

    def forward(self,
                x: torch.Tensor,
                s: torch.Tensor) -> torch.Tensor:
        skip = self._skip(x)
        if self.up:
            x = self.upsample(x)

        x = self.c0(x, s)
        x = self.c1(x, s)

        return x + skip


class Encoder(nn.Module):
    def __init__(self,
                 in_ch: int,
                 base: int,
                 structure_code: int):
        super(Encoder, self).__init__()

        self.backbone = nn.Sequential(
            CBR(in_ch, int(base/2), 1, 1, 0, norm="no"),
            ResBlock(int(base/2), base, padding="reflect", down=True),
            ResBlock(base, base*2, padding="reflect", down=True),
            ResBlock(base*2, base*4, padding="reflect", down=True),
            ResBlock(base*4, base*8, padding="reflect", down=True)
        )

        self.structure = nn.Sequential(
            CBR(base*8, base*8, 3, 1, 0, norm="no", padding="reflect"),
            EqualizedConv2d(base*8, structure_code, 1, 1, 0)
        )

        self.texture = nn.Sequential(
            CBR(base*8, base*16, 3, 2, 0, norm="no"),
            CBR(base*16, base*32, 3, 2, 0, norm="no"),
            nn.AdaptiveAvgPool2d(1),
            CBR(base*32, base*32, 1, 1, 0, norm="no")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.backbone(x)

        structure = self.structure(stem)
        texture = torch.flatten(self.texture(stem), 1)

        return structure, texture


class Decoder(nn.Module):
    def __init__(self,
                 base: int,
                 structure_code: int,
                 texture_dim: int):
        super(Decoder, self).__init__()

        self.dec = self._make_dec(base, structure_code, texture_dim)
        self.out = nn.Sequential(
            EqualizedConv2d(base*2, 3, 3, 1, 1),
            nn.Tanh()
        )

    @staticmethod
    def _make_dec(base: int, structure_code: int, texture_dim: int):
        modules = []
        modules.append(StyleResBlock(structure_code, base*2, texture_dim))
        modules.append(StyleResBlock(base*2, base*4, texture_dim))
        modules.append(StyleResBlock(base*4, base*6, texture_dim))
        modules.append(StyleResBlock(base*6, base*8, texture_dim))
        modules.append(StyleResBlock(base*8, base*8, texture_dim, up=True))
        modules.append(StyleResBlock(base*8, base*8, texture_dim, up=True))
        modules.append(StyleResBlock(base*8, base*4, texture_dim, up=True))
        modules.append(StyleResBlock(base*4, base*2, texture_dim, up=True))

        return nn.ModuleList(modules)

    def forward(self,
                x: torch.Tensor,
                s: torch.Tensor) -> torch.Tensor:
        for layer in self.dec:
            x = layer(x, s)

        return self.out(x)


class Discriminator(nn.Module):
    def __init__(self,
                 in_ch=3,
                 base=64,
                 sn=False):
        super(Discriminator, self).__init__()

        self.dis = nn.Sequential(
            CBR(in_ch, base, 1, 1, 0, norm="no", sn=sn),
            ResBlock(base, base, sn=sn, down=True),
            ResBlock(base, base*2, sn=sn, down=True),
            ResBlock(base*2, base*4, sn=sn, down=True),
            ResBlock(base*4, base*4, sn=sn, down=True),
            ResBlock(base*4, base*8, sn=sn, down=True),
            ResBlock(base*8, base*8, sn=sn, down=True)
        )

        if sn:
            self.fc = nn.Sequential(
                spectral_norm(EqualizedLinear(base*8*4*4, base*8)),
                nn.LeakyReLU(),
                spectral_norm(EqualizedLinear(base*8, 1))
            )
        else:
            self.fc = nn.Sequential(
                EqualizedLinear(base*8*4*4, base*8),
                nn.LeakyReLU(),
                EqualizedLinear(base*8, 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dis(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)


class CooccurenceDiscriminator(nn.Module):
    def __init__(self,
                 in_ch=3,
                 base=64,
                 sn=False):
        super(CooccurenceDiscriminator, self).__init__()

        self.dis = nn.Sequential(
            CBR(in_ch, int(base/2), 1, 1, 0, norm="no", sn=sn),
            ResBlock(int(base/2), base, sn=sn, down=True),
            ResBlock(base, base*2, sn=sn, down=True),
            ResBlock(base*2, base*4, sn=sn, down=True),
            ResBlock(base*4, base*6, sn=sn, down=True),
            ResBlock(base*6, base*6, sn=sn, down=True),
            ResBlock(base*6, base*12, sn=sn),
            ResBlock(base*12, base*6, sn=sn, down=True)
        )

        if sn:
            self.linear = nn.Sequential(
                spectral_norm(EqualizedLinear(base*12, base*32)),
                nn.LeakyReLU(),
                spectral_norm(EqualizedLinear(base*32, base*16)),
                nn.LeakyReLU(),
                spectral_norm(EqualizedLinear(base*16, 1))
            )
        else:
            self.linear = nn.Sequential(
                EqualizedLinear(base*12, base*32),
                nn.LeakyReLU(),
                EqualizedLinear(base*32, base*16),
                nn.LeakyReLU(),
                EqualizedLinear(base*16, 1)
            )

    def forward(self,
                x: torch.Tensor,
                ref: torch.Tensor,
                ref_batch=None,
                r=None) -> (torch.Tensor, torch.Tensor):
        x = self.dis(x)

        if r is None:
            r = self.dis(ref)
            _, ch, h, w = r.size()
            r = r.view(-1, ref_batch, ch, h, w)
            r = r.mean(1)

        x = torch.cat([x, r], dim=1)
        x = torch.flatten(x, 1)

        return self.linear(x), r
