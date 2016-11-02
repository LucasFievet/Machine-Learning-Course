"""Description of this file."""


import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import curve_fit

from .load_data import load_sample_input


__author__ = "lfievet"
__copyright__ = "Copyright 2016, Project One"
__credits__ = ["lfievet"]
__license__ = "No License"
__version__ = "1.0"
__maintainer__ = "lfievet"
__email__ = "lfievet@ethz.ch"
__date__ = "12/10/2016"
__status__ = "Production"


def heatmap(data):
    # sample = 1
    # data = load_sample_input(sample)

    for i in range(60, 150, 10):
        heat_map = data[:, :, i]
        # print(heat_map)
        # print(heat_map.shape)
        # heatmap.reshape((176, 208))
        # print(heatmap.shape)
        sum = np.sum(heat_map)
        # print(sum)

        if sum == 0:
            continue

        # heatmap = np.array((176, 208))
        #
        # for

        # heatmap = data[:, :, 0]
        # # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        #
        plt.clf()
        plt.imshow(heat_map)
        plt.savefig("plots/brain-diff-{}.pdf".format(i))


def double_normal(x, a1, s1, mu1, a2, s2, mu2):
    x1 = x - mu1
    x2 = x - mu2
    return a1*np.exp(-s1*x1*x1) + a2*np.exp(-s2*x2*x2)
