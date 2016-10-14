"""Description of this file."""

import numpy as np

from .load_data import load_samples_inputs

__author__ = "lfievet"
__copyright__ = "Copyright 2016, Project One"
__credits__ = ["lfievet"]
__license__ = "No License"
__version__ = "1.0"
__maintainer__ = "lfievet"
__email__ = "lfievet@ethz.ch"
__date__ = "12/10/2016"
__status__ = "Production"

def feature_ratio_mean(inputs):
    inputs = get_flat_values(inputs)

    fs = []
    for i in inputs:
        mean = np.mean(i)
        low = i[i < mean]
        high = i[i >= mean]

        ratio = np.sum(high)/np.sum(low)

        fs.append(ratio)
    return fs

def feature_mean(inputs):
    inputs = get_flat_values(inputs)

    fs = []
    for i in inputs:
        fs.append(np.mean(i))

    return fs


def feature_ratio(inputs):
    inputs = get_flat_values(inputs)

    fs = []
    for i in inputs:

        low = i[i < 1100]
        high = i[i >= 1100]

        ratio = np.sum(high)/np.sum(low)

        fs.append(ratio)

    return fs


def feature_max(inputs):
    inputs = get_flat_values(inputs)

    fs = []
    for i in inputs:
        hist, bin_edges = np.histogram(i, 50)
        idx_max = hist.argmax()
        x_max = bin_edges[idx_max]

        fs.append(x_max)

    return fs


def get_flat_values(inputs):
#    if inputs is None:
#       inputs = load_samples_inputs()

    values = []
    for i in inputs:
        vs = i.flatten()
        values.append(vs[vs > 100])

    return values
