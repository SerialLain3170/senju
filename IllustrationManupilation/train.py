import yaml
import torch
import torch.nn as nn
import argparse
import pprint

from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
from torch.utils.data import DataLoader

from model import Encoder, Decoder
from model import Discriminator, CooccurenceDiscriminator
from dataset import DanbooruFacesDataset
from visualize import Visualizer
from loss import SwappingLossCalculator
from utils import session, patch_convert


class Trainer:
    def __init__(self,
                 config,
                 outdir,
                 modeldir,
                 data_path):

        self.train_config = config["train"]
        self.data_config = config["dataset"]
        model_config = config["model"]
        self.loss_config = config["loss"]

        self.outdir = outdir
        self.modeldir = modeldir

        self.dataset = DanbooruFacesDataset(data_path,
                                            self.data_config["train_size"],
                                            self.data_config["extension"])
        print(self.dataset)

        enc = Encoder(model_config["encoder"]["in_ch"],
                      model_config["encoder"]["base"],
                      model_config["encoder"]["structure_code"])
        self.enc, self.enc_opt = self._setting_model_optim(enc,
                                                           model_config["encoder"])

        dec = Decoder(model_config["decoder"]["base"],
                      model_config["decoder"]["structure_code"],
                      model_config["decoder"]["texture_dim"])
        self.dec, self.dec_opt = self._setting_model_optim(dec,
                                                           model_config["decoder"])

        dis = Discriminator(sn=model_config["discriminator"]["sn"])
        self.dis, self.dis_opt = self._setting_model_optim(dis,
                                                           model_config["discriminator"])

        con_dis = CooccurenceDiscriminator(sn=model_config["con_discriminator"]["sn"])
        self.con_dis, self.con_dis_opt = self._setting_model_optim(con_dis,
                                                                   model_config["con_discriminator"])

        self.lossfunc = SwappingLossCalculator()
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
                       validsize: int) -> List[torch.Tensor]:

        c_val = dataset.valid(validsize)

        return c_val

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
              validsize: int,
              v_list: List[torch.Tensor]):
        torch.save(self.enc.state_dict(),
                   f"{self.modeldir}/encoder_{iteration}.pt")
        torch.save(self.dec.state_dict(),
                   f"{self.modeldir}/decoder_{iteration}.pt")
        torch.save(self.dis.state_dict(),
                   f"{self.modeldir}/discriminator_{iteration}.pt")
        torch.save(self.con_dis.state_dict(),
                   f"{self.modeldir}/cooccurence_discriminator_{iteration}.pt")

        x_val, s_val = v_list.chunk(2, dim=0)

        with torch.no_grad():
            s0, t0 = self.enc(x_val)
            _, t1 = self.enc(s_val)
            y = self.dec(s0, t1)
            y_recon = self.dec(s0, t0)

        self.visualizer(x_val, s_val, y,
                        self.outdir, iteration, int(validsize/2), "swap")
        self.visualizer(x_val, s_val, y_recon,
                        self.outdir, iteration, int(validsize/2), "recon")

    def _iter(self, data, iteration):
        t = data
        t = t.cuda()

        loss = {}

        t0, t1 = t.chunk(2, dim=0)
        str0, tex0 = self.enc(t0)
        _, tex1 = self.enc(t1)

        y0 = self.dec(str0, tex0)
        y1 = self.dec(str0, tex1)

        y = torch.cat([y0, y1], dim=0)
        dis_adv_loss = self.loss_config["adv"] * self.lossfunc.adversarial_disloss(self.dis,
                                                                                   y.detach(),
                                                                                   t)

        patch_y1 = patch_convert(y1, self.train_config["n_crop"])
        patch_t1 = patch_convert(t1, self.train_config["n_crop"])
        patch_ref = patch_convert(t1, self.train_config["n_crop"] * self.train_config["n_ref"])
        dis_con_loss = self.loss_config["con_adv"] * self.lossfunc.cooccur_hingedis(self.con_dis,
                                                                                    patch_y1.detach(),
                                                                                    patch_t1,
                                                                                    patch_ref,
                                                                                    self.train_config["n_ref"])

        dis_loss = dis_adv_loss + dis_con_loss

        self.dis_opt.zero_grad()
        self.con_dis_opt.zero_grad()
        dis_loss.backward()
        self.dis_opt.step()
        self.con_dis_opt.step()

        if iteration % self.loss_config["d_reg"]["interval"] == 0:
            t.requires_grad = True
            patch_t1.requires_grad = True

            r1_loss, con_r1_loss, d_reg_loss = self.lossfunc.d_regularize(self.dis,
                                                                          self.con_dis,
                                                                          t,
                                                                          patch_t1,
                                                                          patch_ref,
                                                                          self.train_config["n_ref"])
            d_reg_loss += self.loss_config["d_reg"]["w_dis"] / 2 * r1_loss * self.loss_config["d_reg"]["interval"]
            d_reg_loss += self.loss_config["d_reg"]["w_con"] / 2 * con_r1_loss * self.loss_config["d_reg"]["interval"]

            self.dis_opt.zero_grad()
            self.con_dis_opt.zero_grad()
            d_reg_loss.backward()
            self.dis_opt.step()
            self.con_dis_opt.step()
        else:
            d_reg_loss = torch.tensor(0.0).cuda()

        t.requires_grad = False

        str0, tex0 = self.enc(t0)
        _, tex1 = self.enc(t1)

        y0 = self.dec(str0, tex0)
        y1 = self.dec(str0, tex1)

        y = torch.cat([y0, y1], dim=0)

        content_loss = self.loss_config["content"] * self.lossfunc.content_loss(y0, t0)
        gen_adv_loss = self.loss_config["adv"] * self.lossfunc.adversarial_genloss(self.dis,
                                                                                   y)

        patch_y1 = patch_convert(y1, self.train_config["n_crop"])
        patch_ref = patch_convert(t1, self.train_config["n_crop"] * self.train_config["n_ref"])
        gen_con_loss = self.loss_config["con_adv"] * self.lossfunc.cooccur_hingegen(self.con_dis,
                                                                                    patch_y1,
                                                                                    patch_ref,
                                                                                    self.train_config["n_ref"])

        gen_loss = content_loss + gen_adv_loss + gen_con_loss
        self.enc_opt.zero_grad()
        self.dec_opt.zero_grad()
        gen_loss.backward()
        self.enc_opt.step()
        self.dec_opt.step()

        loss["loss_adv_dis"] = dis_adv_loss.item()
        loss["loss_con_dis"] = dis_con_loss.item()
        loss["loss_adv_gen"] = gen_adv_loss.item()
        loss["loss_con_gen"] = gen_con_loss.item()
        loss["loss_content"] = content_loss.item()
        loss["loss_reg"] = d_reg_loss.item()

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
                    loss_dict = self._iter(data, iteration)
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
    parser = argparse.ArgumentParser(description="SwappingAutoencoder")
    parser.add_argument('--session', type=str, default='swapping', help="session name")
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
