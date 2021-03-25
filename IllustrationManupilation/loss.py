import torch
import torch.nn as nn

from torch import autograd

maeloss = nn.L1Loss()
mseloss = nn.MSELoss()
softplus = nn.Softplus()


class SwappingLossCalculator:
    def __init__(self):
        pass

    @staticmethod
    def content_loss(y: torch.Tensor,
                     t: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(y-t))

    @staticmethod
    def adversarial_disloss(discriminator: nn.Module,
                            y: torch.Tensor,
                            t: torch.Tensor) -> torch.Tensor:
        fake = discriminator(y)
        real = discriminator(t)

        loss = torch.mean(softplus(-real)) + torch.mean(softplus(fake))

        return loss

    @staticmethod
    def adversarial_genloss(discriminator: nn.Module,
                            y: torch.Tensor) -> torch.Tensor:
        fake = discriminator(y)

        loss = torch.mean(softplus(-fake))

        return loss

    @staticmethod
    def adversarial_hingedis(discriminator: nn.Module,
                             y: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:
        fake = discriminator(y)
        real = discriminator(t)

        fake_loss = nn.ReLU()(1.0 + fake).mean()
        real_loss = nn.ReLU()(1.0 - real).mean()

        return fake_loss + real_loss

    @staticmethod
    def adversarial_hingegen(discriminator: nn.Module,
                             y: torch.Tensor) -> torch.Tensor:
        fake = discriminator(y)

        return -fake.mean()

    @staticmethod
    def cooccur_hingedis(discriminator: nn.Module,
                         y: torch.Tensor,
                         t: torch.Tensor,
                         ref: torch.Tensor,
                         n_crop: int):

        fake, r_input = discriminator(y, ref, ref_batch=n_crop)
        real, _ = discriminator(t, ref, r=r_input)

        return torch.mean(softplus(-real)) + torch.mean(softplus(fake))

    @staticmethod
    def cooccur_hingegen(discriminator: nn.Module,
                         y: torch.Tensor,
                         ref: torch.Tensor,
                         n_crop: int):

        fake, _ = discriminator(y, ref, ref_batch=n_crop)

        return torch.mean(softplus(-fake))

    @staticmethod
    def _grad_penalty(t_dis: torch.Tensor,
                      t: torch.Tensor) -> torch.Tensor:

        (grad_real,) = autograd.grad(
            outputs=t_dis.sum(), inputs=t, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    def d_regularize(self,
                     discriminator: nn.Module,
                     con_discriminator: nn.Module,
                     t: torch.Tensor,
                     t_patch: torch.Tensor,
                     ref_patch: torch.Tensor,
                     n_crop: int):

        t_dis = discriminator(t)
        r1_loss = self._grad_penalty(t_dis, t)

        t_patch_dis, _ = con_discriminator(t_patch,
                                           ref_patch,
                                           ref_batch=n_crop)
        patch_r1_loss = self._grad_penalty(t_patch_dis, t_patch)

        loss = 0 * t_dis[0, 0] + 0 * t_patch_dis[0, 0]

        return r1_loss, patch_r1_loss, loss
