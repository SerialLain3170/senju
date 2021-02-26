import torch
import numpy as np
import cv2 as cv

from typing import List
from torch.utils.data import Dataset
from pathlib import Path


class BuildDataset(Dataset):
    def __init__(self,
                 data_path: Path,
                 train_size: int,
                 extension=".png"):

        self.data_path = data_path
        self.train_size = train_size

        self.pathlist = list(self.data_path.glob(f"**/*{extension}"))
        self.train_list, self.val_list = self._train_val_split(self.pathlist)

        self.interpolations = (
            cv.INTER_LINEAR,
            cv.INTER_AREA,
            cv.INTER_NEAREST,
            cv.INTER_CUBIC,
            cv.INTER_LANCZOS4
        )

    def __repr__(self):
        return f"The number of dataset is {len(self.train_list)}"

    def __len__(self):
        return len(self.train_list)

    @staticmethod
    def _train_val_split(pathlist: List) -> (List, List):
        split_point = int(len(pathlist) * 0.95)
        train = pathlist[:split_point]
        val = pathlist[split_point:]

        return train, val

    @staticmethod
    def _random_flip(img):
        if np.random.randint(2):
            img = img[:, ::-1, :]

        return img

    @staticmethod
    def _random_crop(image, size):
        height, width = image.shape[0], image.shape[1]
        rnd0 = np.random.randint(height - size - 1)
        rnd1 = np.random.randint(width - size - 1)

        cropped = image[rnd0: rnd0 + size, rnd1: rnd1 + size]

        return cropped

    @staticmethod
    def _totensor(array_list: List) -> torch.Tensor:
        return torch.cuda.FloatTensor(np.array(array_list).astype(np.float32))

    def _down_sample(self, img):
        inter = np.random.choice(self.interpolations)
        img = cv.resize(img, (64, 64), interpolation=inter)

        return img

    def _preprocess(self, img):
        img = self._random_crop(img, self.train_size)
        img = self._random_flip(img)
        l_img = self._down_sample(img)

        img = img[:, :, ::-1].astype(np.float32)
        img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        l_img = l_img[:, :, ::-1].astype(np.float32)
        l_img = (l_img.transpose(2, 0, 1) - 127.5) / 127.5

        return img, l_img

    def valid(self, validsize: int):
        h_valid_box = []
        l_valid_box = []

        for index in range(validsize):
            img_path = self.val_list[index]
            img = cv.imread(str(img_path))
            img, l_img = self._preprocess(img)

            h_valid_box.append(img)
            l_valid_box.append(l_img)

        img = self._totensor(h_valid_box)
        l_img = self._totensor(l_valid_box)

        return img, l_img

    def __getitem__(self, idx):
        img_path = self.train_list[idx]
        img = cv.imread(str(img_path))
        img, l_img = self._preprocess(img)

        return img, l_img
