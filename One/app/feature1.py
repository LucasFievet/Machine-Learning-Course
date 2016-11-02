"""Description of this file."""

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

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


def feature_fit(inputs):
    inputs = get_flat_values(inputs)

    fs = []
    for i in inputs:
        fs.append(np.mean(i))

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

    fs = pd.DataFrame()
    for idx, i in enumerate(inputs):
        y, x = np.histogram(i, 50)

        try:
            x = (x[1:] + x[:-1])/2
            params, covariances = curve_fit(double_normal, x, y, [1E-3, 1E-4, 800, 1E-3, 1E-4, 1400])

            # idx_max = hist.argmax()
            # x_max = bin_edges[idx_max]
            fs[idx] = params

            # fs.append(params[2])
        except:
            fs[idx] = [0, 0, 0, 0, 0, 0]

    fs = fs.transpose()

    # fs_ok = fs[fs[0] > 0]
    fs = fs / fs.mean()

    return fs


def get_flat_values(inputs):
    if inputs is None:
        inputs = load_samples_inputs()

    values = []
    for i in inputs:
        vs = i.get_data().flatten()
        vs = vs[vs > 100]
        vs = vs[vs < 1600]
        values.append(vs[vs > 100])

    return values


def double_normal(x, a1, s1, mu1, a2, s2, mu2):
    x1 = x - mu1
    x2 = x - mu2
    return a1*np.exp(-s1*x1*x1) + a2*np.exp(-s2*x2*x2)
