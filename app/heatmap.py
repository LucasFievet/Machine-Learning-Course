"""Description of this file."""


import matplotlib.pyplot as plt

import numpy as np

from .load_data import load_data


__author__ = "lfievet"
__copyright__ = "Copyright 2016, Project One"
__credits__ = ["lfievet"]
__license__ = "No License"
__version__ = "1.0"
__maintainer__ = "lfievet"
__email__ = "lfievet@ethz.ch"
__date__ = "12/10/2016"
__status__ = "Production"


def heatmap():
    data = load_data()

    for i in range(0, 176):
        heat_map = data.get_data()[:, :, i, 0]
        print(heat_map)
        print(heat_map.shape)
        # heatmap.reshape((176, 208))
        # print(heatmap.shape)
        sum = np.sum(heat_map)
        print(sum)

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
        plt.savefig("plots/test{}.pdf".format(i))
