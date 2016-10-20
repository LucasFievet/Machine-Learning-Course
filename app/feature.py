"""Description of this file."""

import numpy as np

from .normalize import normalize

def feature_ratio_mean(inputs, norm=None):
    inputs = get_flat_values(inputs)

    fs = []
    for i in inputs:
        mean = np.mean(i)
        low = i[i < mean]
        high = i[i >= mean]

        ratio = np.sum(high)/np.sum(low)

        fs.append(ratio)

    if norm == None:
        fs, minmax = normalize(fs)
        return fs, minmax
    else:
        fs = normalize(fs, norm)
        return fs

def feature_mean(inputs, norm=None):
    inputs = get_flat_values(inputs)

    fs = []
    for i in inputs:
        fs.append(np.mean(i))

    if norm == None:
        fs, minmax = normalize(fs)
        return fs, minmax
    else:
        fs = normalize(fs, norm)
        return fs

def feature_max(inputs, norm=None):
    inputs = get_flat_values(inputs)

    fs = []
    for i in inputs:
        hist, bin_edges = np.histogram(i, 50)
        idx_max = hist.argmax()
        x_max = bin_edges[idx_max]

        fs.append(x_max)
    if norm == None:
        fs, minmax = normalize(fs)
        return fs, minmax
    else:
        fs = normalize(fs, norm)
        return fs


def get_flat_values(inputs):
    values = []
    for i in inputs:
        vs = i.flatten()
        values.append(vs[vs > 100])

    return values
