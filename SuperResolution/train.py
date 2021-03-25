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

from loss import RAMCalculator
from model import Generator, Vgg19
from dataset import BuildDataset
from visualize import Visualizer
from utils import session


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

        gen = Generator(num_layers=model_config["generator"]["num_layers"])
        self.gen, self.gen_opt = self._setting_model_optim(gen,
                                                           model_config["generator"])

        self.vgg = Vgg19(requires_grad=False)
        self.vgg.cuda()
        self.vgg.eval()

        self.lossfunc = RAMCalculator()
        self.visualizer = Visualizer()

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
                       ) -> List[torch.Tensor]:

        img, l_img = dataset.valid(validsize)

        return [l_img, img]

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
              v_list: List[torch.Tensor]):
        torch.save(self.gen.state_dict(),
                   f"{self.modeldir}/generator_{iteration}.pt")

        with torch.no_grad():
            y = self.gen(v_list[1])

        self.visualizer(v_list, y, self.outdir, iteration, validsize)

    def _iter(self, data):
        x_h, x_l = data
        x_h = x_h.cuda()
        x_l = x_l.cuda()

        loss = {}

        y = self.gen(x_l)

        con_loss = self.loss_config["content"] * self.lossfunc.content_loss(y, x_h)
        perceptual_loss = self.loss_config["perceptual"] * self.lossfunc.perceptual_loss(self.vgg, y, x_h)

        gen_loss = con_loss + perceptual_loss

        self.gen_opt.zero_grad()
        gen_loss.backward()
        self.gen_opt.step()

        loss["loss_content"] = con_loss.item()
        loss["loss_perceptual"] = perceptual_loss.item()

        return loss

    def __call__(self):
        iteration = 0
        v_list = self._valid_prepare(self.dataset,
                                     self.train_config["validsize"],
                                     )

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
                                   v_list,
                                   )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SuperResolution")
    parser.add_argument('--session', type=str, default='sr', help="session name")
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
