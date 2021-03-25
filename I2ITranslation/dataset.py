import torch
import numpy as np
import cv2 as cv

from copy import copy
from pathlib import Path
from typing import List, Dict
from torch.utils.data import Dataset


class FacesDataset(Dataset):
    def __init__(self,
                 data_path: Path,
                 extension=".png",
                 train_size=128):

        super(FacesDataset, self).__init__()

        self.data_path = data_path
        self.att_list = [path.name for path in self.data_path.iterdir()]
        self.att_map = self._create_hashmap(self.att_list)
        self.att_num = len(self.att_list)

        self.train_list = list(self.data_path.glob(f"**/*{extension}"))
        self.extension = extension
        self.train_size = train_size

    @staticmethod
    def _create_hashmap(att_list: List[str]) -> Dict[str, int]:
        hashmap = {}
        for index, att in enumerate(att_list):
            hashmap[att] = index

        return hashmap

    @staticmethod
    def _label_remove(label_list, source):
        label_list.remove(source)

        return label_list

    @staticmethod
    def _coordinate(img: np.array,
                    color_space="rgb") -> np.array:
        if color_space == "yuv":
            img = img.astype(np.uint8)
            img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
            img = img.transpose(2, 0, 1).astype(np.float32)
            img = (img - 127.5) / 127.5
        elif color_space == "gray":
            img = img.astype(np.uint8)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=0).astype(np.float32)
            img = (img - 127.5) / 127.5
        else:
            img = img[:, :, ::-1].astype(np.float32)
            img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        return img

    @staticmethod
    def _flip(img):
        if np.random.randint(2):
            img = img[:, ::-1, :]

        return img

    @staticmethod
    def _totensor(array_list):
        return torch.cuda.FloatTensor(np.array(array_list).astype(np.float32))

    def _onehot_convert(self, label, path):
        onehot = np.zeros(self.att_num).astype(np.float32)
        onehot[self.att_map[label]] = 1.0

        return onehot

    def _get_src_att(self, label_list):
        rnd_att = np.random.choice(label_list)
        pathlist = list((self.data_path / Path(str(rnd_att))).glob(f"*{self.extension}"))

        img_path = np.random.choice(pathlist)
        onehot = self._onehot_convert(rnd_att, img_path)

        img = cv.imread(str(img_path))
        img = cv.resize(img, (self.train_size, self.train_size))
        img = self._flip(img)
        img = self._coordinate(img)

        return rnd_att, img, onehot

    def _preprocess(self):
        att_list = copy(self.att_list)
        rnd_att, x_src, x_c = self._get_src_att(att_list)

        att_list = self._label_remove(att_list, rnd_att)
        rnd_att, y_src, y_c = self._get_src_att(att_list)

        att_list = self._label_remove(att_list, rnd_att)
        _, z_src, z_c = self._get_src_att(att_list)

        return x_src, x_c, y_src, y_c, z_src, z_c

    def __repr__(self):
        return f"dataset length: {len(self.train_list)}"

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        return self._preprocess()
