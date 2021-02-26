import torch
import torch.nn as nn
import numpy as np

from utils import call_zeros
from torch import autograd
from torch.autograd import Variable

maeloss = nn.L1Loss()
mseloss = nn.MSELoss()
softplus = nn.Softplus()


class CategoricalLoss(nn.Module):
    def __init__(self, atoms=51, v_max=10, v_min=-10):
        super(CategoricalLoss, self).__init__()

        self.atoms = atoms
        self.v_max = v_max
        self.v_min = v_min
        self.supports = torch.linspace(v_min, v_max, atoms).view(1, 1, atoms).cuda() # RL: [bs, #action, #quantiles]
        self.delta = (v_max - v_min) / (atoms - 1)

    def forward(self, anchor, feature, skewness=0.0):
        batch_size = feature.shape[0]
        skew = torch.zeros((batch_size, self.atoms)).cuda().fill_(skewness)

        # experiment to adjust KL divergence between positive/negative anchors
        Tz = skew + self.supports.view(1, -1) * torch.ones((batch_size, 1)).to(torch.float).view(-1, 1).cuda()
        Tz = Tz.clamp(self.v_min, self.v_max)
        b = (Tz - self.v_min) / self.delta
        l = b.floor().to(torch.int64)
        u = b.ceil().to(torch.int64)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.atoms - 1)) * (l == u)] += 1
        offset = torch.linspace(0, (batch_size - 1) * self.atoms, batch_size).to(torch.int64).unsqueeze(dim=1).expand(batch_size, self.atoms).cuda()
        skewed_anchor = torch.zeros(batch_size, self.atoms).cuda()
        skewed_anchor.view(-1).index_add_(0, (l + offset).view(-1), (anchor * (u.float() - b)).view(-1))  
        skewed_anchor.view(-1).index_add_(0, (u + offset).view(-1), (anchor * (b - l.float())).view(-1))  

        loss = -(skewed_anchor * (feature + 1e-16).log()).sum(-1).mean()

        return loss


class LossCalculator:
    def __init__(self):
        self.triplet = CategoricalLoss()

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

        loss = nn.ReLU()(1.0 + fake).mean()
        loss += nn.ReLU()(1.0 - real).mean()

        return loss

    @staticmethod
    def adversarial_hingegen(discriminator: nn.Module,
                             y: torch.Tensor) -> torch.Tensor:

        fake = discriminator(y)
        loss = -fake.mean()

        return loss

    @staticmethod
    def adversarial_relativistic_disloss(discriminator: nn.Module,
                                         y: torch.Tensor,
                                         t: torch.Tensor) -> torch.Tensor:
        fake = discriminator(y)
        real = discriminator(t)

        loss = torch.mean(softplus(-(real - fake)))

        return loss

    @staticmethod
    def adversarial_relativistic_genloss(discriminator: nn.Module,
                                         y: torch.Tensor,
                                         t: torch.Tensor) -> torch.Tensor:

        fake = discriminator(y)
        real = discriminator(t).detach()

        loss = torch.mean(softplus(-(fake - real)))

        return loss

    @staticmethod
    def adversarial_relativistic_average_disloss(discriminator: nn.Module,
                                                 y: torch.Tensor,
                                                 t: torch.Tensor) -> torch.Tensor:
        fake = discriminator(y)
        real = discriminator(t)

        fake_mean = fake.mean(0, keepdims=True)
        real_mean = real.mean(0, keepdims=True)

        loss = torch.mean(softplus(-(real - fake_mean))) + torch.mean(softplus(fake - real_mean))

        return loss

    @staticmethod
    def adversarial_relativistic_average_genloss(discriminator: nn.Module,
                                                 y: torch.Tensor,
                                                 t: torch.Tensor) -> torch.Tensor:

        fake = discriminator(y)
        real = discriminator(t).detach()

        fake_mean = fake.mean(0, keepdims=True)
        real_mean = real.mean(0, keepdims=True)

        loss = torch.mean(softplus(-(fake - real_mean))) + torch.mean(softplus(real - fake_mean))

        return loss

    def realness_disloss(self,
                         discriminator: nn.Module,
                         y: torch.Tensor,
                         t: torch.Tensor,
                         anchor1: torch.Tensor,
                         anchor0: torch.Tensor) -> torch.Tensor:
        outcomes = self.triplet.atoms

        anc_real = call_zeros(t, outcomes, anchor1)
        anc_fake = call_zeros(t, outcomes, anchor0)

        t_dis = discriminator(t).log_softmax(1).exp()
        y_dis = discriminator(y).log_softmax(1).exp()

        loss_real = self.triplet(anc_real, t_dis, skewness=1.0)
        loss_fake = self.triplet(anc_fake, y_dis, skewness=-1.0)

        return loss_real + loss_fake

    def realness_genloss(self,
                         discriminator: nn.Module,
                         y: torch.Tensor,
                         t: torch.Tensor,
                         anchor1: torch.Tensor,
                         anchor0: torch.Tensor) -> torch.Tensor:
        outcomes = self.triplet.atoms

        anc_real = call_zeros(t, outcomes, anchor1)
        anc_fake = call_zeros(t, outcomes, anchor0)

        t_dis = discriminator(t).log_softmax(1).exp()
        y_dis = discriminator(y).log_softmax(1).exp().detach()

        loss_real = -self.triplet(anc_fake, y_dis, skewness=-1.0)
        loss_fake = self.triplet(t_dis, y_dis)

        return loss_real + loss_fake

    @staticmethod
    def gradient_penalty(discriminator: nn.Module,
                         y: torch.Tensor,
                         t: torch.Tensor,
                         center="zero",
                         gp_type="wgangp") -> torch.Tensor:

        alpha = torch.cuda.FloatTensor(np.random.random(size=t.shape))
        epsilon = torch.rand(t.size()).cuda()

        if gp_type == "wgangp":
            interpolates = alpha * t + (1 - alpha) * y
        elif gp_type == "dragan":
            interpolates = alpha * t + ((1 - alpha) * (t + 0.5 * t.std() * epsilon))
        interpolates = Variable(interpolates, requires_grad=True)

        d_interpolates = discriminator(interpolates)

        fake = Variable(torch.cuda.FloatTensor(t.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False)

        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        if center == "one":
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        elif center == "zero":
            gradient_penalty = ((gradients.norm(2, dim=1)) ** 2).mean()

        return gradient_penalty
