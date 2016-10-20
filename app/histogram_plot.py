"""Description of this file."""


import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import curve_fit


__author__ = "lfievet"
__copyright__ = "Copyright 2016, Project One"
__credits__ = ["lfievet"]
__license__ = "No License"
__version__ = "1.0"
__maintainer__ = "lfievet"
__email__ = "lfievet@ethz.ch"
__date__ = "20/10/2016"
__status__ = "Production"


def histogram_plot(flat_data, filename):
    # flat_data = flat_data[flat_data > 2]
    # flat_data = flat_data[flat_data < 1600]

    plt.figure()
    y, x, z = plt.hist(flat_data, 50, normed=1, facecolor='green', alpha=0.75)
    # x = (x[1:] + x[:-1])/2

    # xs = np.linspace(100, 1600, 100)
    # popt, pcov = curve_fit(double_normal, x, y, [1E-3, 1E-4, 800, 1E-3, 1E-4, 1400])
    # print(popt)
    # print(pcov)
    # ys = double_normal(xs, *popt)

    # plt.plot(xs, ys, "k-")

    plt.xlabel('Intensity')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ Intensity}$')
    # plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    plt.savefig("plots/{}.pdf".format(filename))


def double_normal(x, a1, s1, mu1, a2, s2, mu2):
    x1 = x - mu1
    x2 = x - mu2
    return a1*np.exp(-s1*x1*x1) + a2*np.exp(-s2*x2*x2)
