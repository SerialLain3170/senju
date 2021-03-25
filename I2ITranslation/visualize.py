import numpy as np

import matplotlib
matplotlib.use("Agg")
import pylab


class Visualizer:
    def __init__(self):
        pylab.rcParams['figure.figsize'] = (16.0, 16.0)
        pylab.clf()

    @staticmethod
    def _coordinate(array):
        tmp = np.clip(array*127.5+127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)

        return tmp

    def onesave(self, y, num, outdir):
        pylab.rcParams['figure.figsize'] = (16.0, 16.0)
        pylab.clf()
        for index in range(16):
            tmp = self._coordinate(y[index])
            pylab.subplot(4, 4, index + 1)
            pylab.imshow(tmp)
            pylab.axis("off")
            pylab.savefig(f"./gifdif/visualize_{num}.png")

    def __call__(self, y, x, testsize, nc_size, color_num, flag):
        if flag:
            for index in range(testsize):
                tmp = self._coordinate(x[index])
                pylab.subplot(testsize, 1 + nc_size, (1 + nc_size) * index + 1)
                pylab.imshow(tmp)
                pylab.axis("off")
                pylab.savefig(f"visualize.png")

        for index in range(testsize):
            tmp = self._coordinate(y[index])
            pylab.subplot(testsize, 1 + nc_size, (1 + nc_size) * index + color_num + 2)
            pylab.imshow(tmp)
            pylab.axis("off")
            pylab.savefig(f"visualize.png")