import numpy as np
import cv2 as cv

from torch.utils.data import Dataset
from pathlib import Path


class BuildDataset(Dataset):
    def __init__(self,
                 data_path: Path,
                 train_size: int,
                 extension=".png"):

        self.data_path = data_path
        self.train_size = train_size

        self.pathlist = list(self.data_path.glob(f"*{extension}"))

    def __repr__(self):
        return f"The number of dataset is {len(self.pathlist)}"

    def __len__(self):
        return len(self.pathlist)

    @staticmethod
    def _random_flip(img):
        if np.random.randint(2):
            img = img[:, ::-1, :]

        return img

    def _preprocess(self, img):
        img = cv.resize(img,
                        (self.train_size, self.train_size),
                        interpolation=cv.INTER_CUBIC)
        img = self._random_flip(img)
        img = img[:, :, ::-1].astype(np.float32)
        img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        return img

    def __getitem__(self, idx):
        img_path = self.pathlist[idx]
        img = cv.imread(img_path)
        img = self._preprocess(img)

        return img
