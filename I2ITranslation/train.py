import torch
import yaml
import pprint
import argparse
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from dataset import FacesDataset
from model import Generator, Discriminator
from loss import RelGANLossCalculator
from utils import session


class Trainer:
    def __init__(self,
                 config,
                 modeldir,
                 data_path,):

        self.train_config = config["train"]
        self.data_config = config["dataset"]
        model_config = config["model"]
        self.loss_config = config["loss"]

        self.modeldir = modeldir

        self.dataset = FacesDataset(data_path,
                                    self.data_config["extension"],
                                    self.data_config["train_size"])
        print(self.dataset)

        gen = Generator(nc_size=self.dataset.att_num,
                        layers=model_config["generator"]["layers"])
        self.gen, self.gen_opt = self._setting_model_optim(gen,
                                                           model_config["generator"])

        dis = Discriminator(nc_size=self.dataset.att_num)
        self.dis, self.dis_opt = self._setting_model_optim(dis,
                                                           model_config["discriminator"])

        self.lossfunc = RelGANLossCalculator()

    @staticmethod
    def _setting_model_optim(model: nn.Module,
                             config: Dict):
        model.cuda()
        if config["mode"] == "train":
            model.train()
        elif config["mode"] == "eval":
            model.eval()

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config["lr"],
                                     betas=(config["b1"], config["b2"]),
                                     weight_decay=config["wd"])

        return model, optimizer

    @staticmethod
    def _build_dict(loss_dict: Dict[str, float],
                    epoch: int,
                    num_epochs: int) -> Dict[str, str]:

        report_dict = {}
        report_dict["epoch"] = f"{epoch}/{num_epochs}"
        for k, v in loss_dict.items():
            report_dict[k] = f"{v:.4f}"

        return report_dict

    def _eval(self,
              iteration: int,
              ):
        torch.save(self.gen.state_dict(),
                   f"{self.modeldir}/generator_{iteration}.pt")
        torch.save(self.dis.state_dict(),
                   f"{self.modeldir}/discriminator_{iteration}.pt")

    @staticmethod
    def _alpha_generate(batchsize, nc):
        flag = np.random.randint(2)
        if flag:
            alpha = torch.Tensor(batchsize, 1).uniform_(0.5, 1.0)
        else:
            alpha = torch.Tensor(batchsize, 1).uniform_(0.0, 0.5)

        alpha = alpha.cuda()
        alpha = alpha.repeat(1, nc)

        return alpha, flag

    def _iter(self, data):
        x, x_c, t, t_c, z, z_c = data
        x = x.cuda()
        t = t.cuda()
        z = z.cuda()
        x_c = x_c.cuda()
        t_c = t_c.cuda()
        z_c = z_c.cuda()

        loss = {}

        # Discriminator update
        # Adv loss
        a = t_c - x_c
        y = self.gen(x, a)
        adv_dis_loss = self.loss_config["adv"] * self.lossfunc.adversarial_loss_dis(self.dis,
                                                                                    y.detach(),
                                                                                    t)

        # Interpolation loss
        alpha, flag = self._alpha_generate(self.train_config["batchsize"],
                                           self.dataset.att_num)

        y0 = self.gen(x, t_c - t_c)
        y1 = self.gen(x, alpha * a)

        adv_inp_dis_loss = self.loss_config["inp"] * self.lossfunc.interpolation_loss_dis(self.dis,
                                                                                          y0.detach(),
                                                                                          y.detach(),
                                                                                          y1.detach(),
                                                                                          alpha,
                                                                                          flag)

        # Matching loss
        v2 = t_c - z_c
        v3 = z_c - x_c

        adv_mat_dis_loss = self.loss_config["mat"] * self.lossfunc.matching_loss_dis(self.dis, x, y.detach(), t,
                                                                                     z, a, v2, v3)

        dis_loss = adv_dis_loss + adv_inp_dis_loss + adv_mat_dis_loss

        self.dis_opt.zero_grad()
        dis_loss.backward()
        self.dis_opt.step()

        if self.loss_config["gp"] > 0.0:
            t.requires_grad = True

            gp_loss = self.loss_config["gp"] * self.lossfunc.zero_grad_penalty(self.dis,
                                                                               y,
                                                                               t)
            self.dis_opt.zero_grad()
            gp_loss.backward()
            self.dis_opt.step()

            t.requires_grad = False

        # Generator update
        # Adv loss
        y = self.gen(x, a)
        adv_gen_loss = self.loss_config["adv"] * self.lossfunc.adversarial_loss_gen(self.dis,
                                                                                   y)

        # Interpolation loss
        alpha, rnd = self._alpha_generate(self.train_config["batchsize"],
                                          self.dataset.att_num)
        y_alpha = self.gen(x, alpha * a)
        adv_inp_gen_loss = self.loss_config["inp"] * self.lossfunc.interpolation_loss_gen(self.dis,
                                                                                          y_alpha)

        # Mat loss
        adv_mat_gen_loss = self.loss_config["mat"] * self.lossfunc.matching_loss_gen(self.dis,
                                                                                     x, y, a)

        # Cycle-consistency loss
        cyc = self.gen(y, -a)
        cyc_loss = self.loss_config["cycle"] * self.lossfunc.content_loss(cyc, x)

        # Self-reconstruction loss
        rec = self.gen(x, t_c - t_c)
        rec_loss = self.loss_config["recon"] * self.lossfunc.content_loss(rec, x)

        gen_loss = adv_gen_loss + adv_inp_gen_loss + adv_mat_gen_loss + cyc_loss + rec_loss

        self.gen_opt.zero_grad()
        gen_loss.backward()
        self.gen_opt.step()

        loss["loss_adv_dis"] = adv_dis_loss.item()
        loss["loss_inp_dis"] = adv_inp_dis_loss.item()
        loss["loss_mat_dis"] = adv_mat_dis_loss.item()
        loss["loss_adv_gen"] = adv_gen_loss.item()
        loss["loss_inp_gen"] = adv_inp_gen_loss.item()
        loss["loss_mat_gen"] = adv_mat_gen_loss.item()
        loss["loss_cyc"] = cyc_loss.item()
        loss["loss_recon"] = rec_loss.item()
        if self.loss_config["gp"] > 0.0:
            loss["loss_gp"] = gp_loss.item()

        return loss

    def __call__(self):
        iteration = 0

        for epoch in range(self.train_config["epoch"]):
            dataloader = DataLoader(self.dataset,
                                    batch_size=self.train_config["batchsize"],
                                    shuffle=True,
                                    drop_last=True)

            with tqdm(total=len(self.dataset)) as pbar:
                for index, data in enumerate(dataloader):
                    iteration += 1
                    loss_dict = self._iter(data)
                    report_dict = self._build_dict(loss_dict,
                                                   epoch,
                                                   self.train_config["epoch"])

                    pbar.update(self.train_config["batchsize"])
                    pbar.set_postfix(**report_dict)

                    if iteration % self.train_config["snapshot_interval"] == 1:
                        self._eval(iteration,
                                   )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RelGAN")
    parser.add_argument('--session', type=str, default='relgan', help="session name")
    parser.add_argument('--data_path', type=Path, help="path containing color images")
    args = parser.parse_args()

    outdir, modeldir = session(args.session)

    with open("param.yaml", "r") as f:
        config = yaml.safe_load(f)
        pprint.pprint(config)

    trainer = Trainer(config,
                      modeldir,
                      args.data_path,
                      )
    trainer()
