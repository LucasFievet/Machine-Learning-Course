"""Visualizing data."""

import matplotlib.pyplot as plt

import numpy as np

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


def heatmap():
    sample = 10
    data = load_sample_input(sample)

    # flat_data = data.get_data().flatten()
    # flat_data = flat_data[flat_data > 100]
    #
    # plt.hist(flat_data, 50, normed=1, facecolor='green', alpha=0.75)
    # plt.xlabel('Smarts')
    # plt.ylabel('Probability')
    # plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    # # plt.axis([40, 160, 0, 0.03])
    # plt.grid(True)
    # plt.savefig("plots/hist{}.pdf".format(sample))
    #
    # # print(flat_data)
    # return

    for i in range(60, 150):
        heat_map = data.get_data()[:, :, i, 0]
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
        plt.savefig("plots/test{}.pdf".format(i))
