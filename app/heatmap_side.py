"""Description of this file."""


import matplotlib.pyplot as plt

import numpy as np

from .load_data import load_sample_input
from .cut_frontal_lobe import cut_frontal_lobe

def heatmap_side():
    sample = 20
    data = load_sample_input(sample)

    length = len(data.get_data()[:, 1, 1, 0])
    heat_map = data.get_data()[0, :, :, 0]

    for i in range(0, length):
        heat_map += data.get_data()[i, :, :, 0]

    #fig = plt.figure()
    #ax = fig.gca()
    plt.xticks(np.arange(0,len(data.get_data()[0, 0, :, 0]),10))
    plt.yticks(np.arange(0,len(data.get_data()[0, :, 0, 0]),10))
    plt.clf()
    plt.imshow(heat_map)
    plt.grid()
    plt.savefig("plots/avg_side_view.pdf")

    cut_brain()

def cut_brain():
    sample = 20

    data = cut_frontal_lobe(sample)
    heat_map = data[90,:,:]
    plt.clf()
    plt.imshow(heat_map)
    plt.grid()
    plt.savefig("plots/frontal_lobe.pdf")
