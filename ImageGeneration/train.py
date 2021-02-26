import yaml
import torch
import torch.nn as nn
import numpy as np
import argparse
import pprint

from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from loss import LossCalculator
from model import Generator, Discriminator
from dataset import BuildDataset
from visualize import Visualizer
from utils import session, noise_generate


class Trainer:
    def __init__(self,
                 config,
                 outdir,
                 modeldir,
                 data_path,
                 ):

        self.train_config = config["train"]
        self.data_config = config["dataset"]
        model_config = config["model"]
        self.loss_config = config["loss"]

        self.outdir = outdir
        self.modeldir = modeldir

        self.dataset = BuildDataset(data_path,
                                    self.data_config["train_size"],
                                    self.data_config["extension"])
        print(self.dataset)

        gen = Generator(z_dim=model_config["generator"]["z_dim"],
                        sn=model_config["generator"]["sn"])
        self.gen, self.gen_opt = self._setting_model_optim(gen,
                                                           model_config["generator"])

        dis = Discriminator(sn=model_config["discriminator"]["sn"])
        self.dis, self.dis_opt = self._setting_model_optim(dis,
                                                           model_config["discriminator"])

        self.lossfunc = LossCalculator()
        self.visualizer = Visualizer()

        # Anchor0 definition
        gauss = np.random.normal(0, 0.1, 1000)
        count, bins = np.histogram(gauss, 51)
        self.anchor0 = count / sum(count)

        # Anchor1 definition
        unif = np.random.uniform(-1, 1, 1000)
        count, bins = np.histogram(unif, 51)
        self.anchor1 = count / sum(count)

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
                                     betas=(config["b1"], config["b2"]))

        return model, optimizer

    @staticmethod
    def _valid_prepare(dataset,
                       validsize: int,
                       z_dim: int) -> List[torch.Tensor]:

        z = noise_generate(validsize, z_dim)

        return z

    @staticmethod
    def _build_dict(loss_dict: Dict[str, float],
                    epoch: int,
                    num_epochs: int) -> Dict[str, str]:

        report_dict = {}
        report_dict["epoch"] = f"{epoch}/{num_epochs}"
        for k, v in loss_dict.items():
            report_dict[k] = f"{v:.6f}"

        return report_dict

    def _eval(self,
              iteration: int,
              validsize: int,
              z: torch.Tensor):
        torch.save(self.gen.state_dict(),
                   f"{self.modeldir}/generator_{iteration}.pt")
        torch.save(self.dis.state_dict(),
                   f"{self.modeldir}/discriminator_{iteration}.pt")

        with torch.no_grad():
            y = self.gen(z)

        self.visualizer(y, validsize, iteration, self.outdir)

    def _adv_dis_loss(self,
                      discriminator: nn.Module,
                      y: torch.Tensor,
                      t: torch.Tensor) -> torch.Tensor:
        if self.loss_config["adv"]["type"] == "vanilla":
            loss = self.loss_config["adv"]["w"] * self.lossfunc.adversarial_disloss(discriminator,
                                                                                    y,
                                                                                    t)
        elif self.loss_config["adv"]["type"] == "hinge":
            loss = self.loss_config["adv"]["w"] * self.lossfunc.adversarial_hingedis(discriminator,
                                                                                     y,
                                                                                     t)
        elif self.loss_config["adv"]["type"] == "relativistic":
            loss = self.loss_config["adv"]["w"] * self.lossfunc.adversarial_relativistic_disloss(discriminator,
                                                                                                 y,
                                                                                                 t)
        elif self.loss_config["adv"]["type"] == "ra":
            loss = self.loss_config["adv"]["w"] * self.lossfunc.adversarial_relativistic_average_disloss(discriminator,
                                                                                                         y,
                                                                                                         t)
        elif self.loss_config["adv"]["type"] == "realness":
            loss = self.loss_config["adv"]["w"] * self.lossfunc.realness_disloss(discriminator,
                                                                                 y,
                                                                                 t,
                                                                                 self.anchor1,
                                                                                 self.anchor0)

        return loss

    def _adv_gen_loss(self,
                      discriminator: nn.Module,
                      y: torch.Tensor,
                      t: torch.Tensor) -> torch.Tensor:

        if self.loss_config["adv"]["type"] == "vanilla":
            loss = self.loss_config["adv"]["w"] * self.lossfunc.adversarial_genloss(discriminator,
                                                                                    y)
        elif self.loss_config["adv"]["type"] == "hinge":
            loss = self.loss_config["adv"]["w"] * self.lossfunc.adversarial_hingegen(discriminator,
                                                                                     y)
        elif self.loss_config["adv"]["type"] == "relativistic":
            loss = self.loss_config["adv"]["w"] * self.lossfunc.adversarial_relativistic_genloss(discriminator,
                                                                                                 y,
                                                                                                 t)
        elif self.loss_config["adv"]["type"] == "ra":
            loss = self.loss_config["adv"]["w"] * self.lossfunc.adversarial_relativistic_average_genloss(discriminator,
                                                                                                         y,
                                                                                                         t)
        elif self.loss_config["adv"]["type"] == "realness":
            loss = self.loss_config["adv"]["w"] * self.lossfunc.realness_genloss(discriminator,
                                                                                 y,
                                                                                 t,
                                                                                 self.anchor1,
                                                                                 self.anchor0)

        return loss

    def _iter(self, data):
        t = data
        t = t.cuda()

        z = noise_generate(t.size(0), self.train_config["z_dim"])

        loss = {}

        y = self.gen(z)

        # discriminator process
        adv_dis_loss = self._adv_dis_loss(self.dis, y.detach(), t)
        gp_loss = self.loss_config["gp"]["w"] * self.lossfunc.gradient_penalty(self.dis,
                                                                          y.detach(),
                                                                          t,
                                                                          center=self.loss_config["gp"]["center"],
                                                                          gp_type=self.loss_config["gp"]["type"])

        dis_loss = adv_dis_loss + gp_loss

        self.dis_opt.zero_grad()
        dis_loss.backward()
        self.dis_opt.step()

        # generator process
        z = noise_generate(t.size(0), self.train_config["z_dim"])
        y = self.gen(z)

        adv_gen_loss = self._adv_gen_loss(self.dis, y, t)

        gen_loss = adv_gen_loss

        self.gen_opt.zero_grad()
        gen_loss.backward()
        self.gen_opt.step()

        loss["loss_adv_dis"] = adv_dis_loss.item()
        loss["loss_gp"] = gp_loss.item()
        loss["loss_adv_gen"] = adv_gen_loss.item()

        return loss

    def __call__(self):
        iteration = 0
        z_fix = self._valid_prepare(self.dataset,
                                    self.train_config["validsize"],
                                    self.train_config["z_dim"])

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
                                   self.train_config["validsize"],
                                   z_fix,
                                   )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ImageGeneration")
    parser.add_argument('--session', type=str, default='gan', help="session name")
    parser.add_argument('--data_path', type=Path, help="path containing color images")
    args = parser.parse_args()

    outdir, modeldir = session(args.session)

    with open("param.yaml", "r") as f:
        config = yaml.safe_load(f)
        pprint.pprint(config)

    trainer = Trainer(config,
                      outdir,
                      modeldir,
                      args.data_path,
                      )
    trainer()
