import torch
import numpy as np
import cv2 as cv

from torch.utils.data import Dataset
from pathlib import Path
from typing import List


class DanbooruFacesDataset(Dataset):
    def __init__(self,
                 data_path: Path,
                 train_size: int,
                 extension=".jpg"):
        super(DanbooruFacesDataset, self).__init__()

        self.data_path = data_path
        self.pathlist = list(self.data_path.glob(f'**/*{extension}'))
        self.train_list, self.val_list = self._train_val_split(self.pathlist)
        self.train_len = len(self.train_list)

        self.train_size = train_size

    @staticmethod
    def _train_val_split(pathlist: List[Path]) -> (List, List):
        split_point = int(len(pathlist) * 0.9)
        train = pathlist[:split_point]
        val = pathlist[split_point:]

        return train, val

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

    def _preprocess(self, color, d_type="getchu"):
        if d_type == "getchu":
            h, w = color.shape[0], color.shape[1]
            color = color[:h-50, 25:w-25, :]
        color = cv.resize(color,
                          (self.train_size, self.train_size),
                          interpolation=cv.INTER_CUBIC)
        color = self._coordinate(color)

        return color

    def valid(self, validsize, mode="straight"):
        c_valid_box = []

        for index in range(validsize):
            if mode == "straight":
                color_path = self.val_list[index]
            elif mode == "random":
                color_path = np.random.choice(self.val_list)
            color = cv.imread(str(color_path))
            color = self._preprocess(color, d_type="getchu")

            c_valid_box.append(color)

        color = self._totensor(c_valid_box)

        return color

    @staticmethod
    def _totensor(array_list):
        return torch.cuda.FloatTensor(np.array(array_list).astype(np.float32))

    def __repr__(self):
        return f"dataset length: {self.train_len}"

    def __len__(self):
        return self.train_len

    def __getitem__(self, idx):
        color_path = self.train_list[idx]
        color = cv.imread(str(color_path))

        color = self._preprocess(color, d_type="danbooru")

        return color
