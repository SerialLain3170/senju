import numpy as np

import matplotlib
import pylab


class Visualizer:
    def __init__(self):
        pass

    @staticmethod
    def _convert(img):
        img = img.detach().cpu().numpy()
        img = np.clip(img*127.5 + 127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)

    def __call__(self, y, validsize, iteration, outdir):
        width = np.sqrt(validsize)
        pylab.rcParams['figure.figsize'] = (16.0, 16.0)
        pylab.clf()

        for index in range(validsize):
            tmp = self._convert(y[index])
            pylab.subplot(width, width, index + 1)
            pylab.imshow(tmp)
            pylab.axis("off")
            pylab.savefig(f"{outdir}/visualize_{iteration}.png")
