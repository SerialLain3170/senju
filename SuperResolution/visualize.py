import numpy as np
import cv2 as cv
import torch

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import pylab


class Visualizer:
    def __init__(self):
        pass

    def _convert(self, img_array: torch.Tensor) -> np.array:
        tmp = img_array.detach().cpu().numpy()
        tmp = np.clip(tmp*127.5 + 127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)

        return tmp

    def _save(self,
              img: torch.Tensor,
              width: int,
              index: int,
              outdir: Path,
              epoch: int):
        tmp = self._convert(img)
        pylab.subplot(width, width, index)
        pylab.imshow(tmp)
        pylab.axis("off")
        pylab.savefig(f"{outdir}/visualize_{epoch}.png")

    def __call__(self,
                 v_list: List[torch.Tensor],
                 y: torch.Tensor,
                 outdir: Path,
                 epoch: int,
                 testsize: int):

        width = int(testsize / 2)
        pylab.rcParams['figure.figsize'] = (16.0, 16.0)
        pylab.clf()

        for index in range(testsize):
            self._save(v_list[0][index], width, 3*index+1, outdir, epoch)
            self._save(v_list[1][index], width, 3*index+2, outdir, epoch)
            self._save(y[index], width, 3*index+3, outdir, epoch)
