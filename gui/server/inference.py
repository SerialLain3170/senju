import yaml
import torch
import numpy as np
import torch.nn as nn
import cv2 as cv
import argparse
import pprint

from pathlib import Path
from typing import List, Dict

from model_stylegan2 import Encoder, Decoder


class Inferer:
    def __init__(self):
        config = self._yaml_load("./param.yaml")
        model_config = config["model"]
        self.enc = self._encoder_setting(model_config)
        self.dec = self._decoder_setting(model_config)

    @staticmethod
    def _yaml_load(yaml_path):
        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        return config

    @staticmethod
    def _encoder_setting(model_config):
        enc = Encoder(model_config["encoder"]["in_ch"],
                      model_config["encoder"]["base"],
                      model_config["encoder"]["structure_code"])
        enc.eval()

        weight = torch.load("./ckpts/encoder_160001.pt", map_location=torch.device("cpu"))
        enc.load_state_dict(weight)

        return enc

    @staticmethod
    def _decoder_setting(model_config):
        dec = Decoder(model_config["decoder"]["base"],
                  model_config["decoder"]["structure_code"],
                  model_config["decoder"]["texture_dim"])
        dec.eval()

        weight = torch.load("./ckpts/decoder_160001.pt", map_location=torch.device("cpu"))
        dec.load_state_dict(weight)

        return dec

    @staticmethod
    def _coordinate(img):
        img = cv.resize(img, (256, 256))
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        img = (img - 127.5) / 127.5
        img = np.expand_dims(img, axis=0)

        return torch.Tensor(img)

    @staticmethod
    def _denorm(img):
        img = img[0].detach().cpu().numpy()
        img = np.clip(img * 127.5 + 127.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)

        return img

    def __call__(self, l, r):
        l = self._coordinate(l)
        r = self._coordinate(r)

        with torch.no_grad():
            l_enc, _ = self.enc(l)
            _, s_enc = self.enc(r)

            y = self.dec(l_enc, s_enc)

        y = self._denorm(y)

        return y
