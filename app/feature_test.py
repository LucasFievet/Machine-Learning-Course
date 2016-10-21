"""Description of this file."""

import numpy as np

from .normalize import normalize
from .get_number_of_files import get_number_of_files
from .load_deviations import load_deviation
from .cut_brain_test import cut_brain

def feature_ratio_mean(area="whole", training=True, norm=None):
    fs = []
    for i in range(1,get_number_of_files(training)+1):
        data = load_deviation(i,training) 
        if not area == "whole":
            data = cut_brain(data,area)
        data = get_flat_values(data) 
        mean = np.mean(data)
        low = data[data < mean]
        high = data[data >= mean]

        ratio = np.sum(high)/np.sum(low)

        fs.append(ratio)

    if norm == None:
        fs, minmax = normalize(fs)
        return fs, minmax
    else:
        fs = normalize(fs, norm)
        return fs

def feature_mean(area, training=True, norm=None):
    fs = []
    for i in range(1,get_number_of_files(training)+1):
        data = load_deviation(i,training) 
        if not area == "whole":
            data = cut_brain(data,area)
        data = get_flat_values(data) 
        fs.append(np.mean(data))

    if norm == None:
        fs, minmax = normalize(fs)
        return fs, minmax
    else:
        fs = normalize(fs, norm)
        return fs

def feature_max(area, training=True, norm=None):
    fs = []
    for i in range(1,get_number_of_files(training)+1):
        data = load_deviation(i,training) 
        if not area == "whole":
            data = cut_brain(data,area)
        data = get_flat_values(data) 
        hist, bin_edges = np.histogram(data, 50)
        idx_max = hist.argmax()
        x_max = bin_edges[idx_max]
        fs.append(x_max)

    print(fs)
    if norm == None:
        fs, minmax = normalize(fs)
        return fs, minmax
    else:
        fs = normalize(fs, norm)
        return fs

def get_flat_values(inputs):
    vs = inputs.flatten()
    return vs #[vs > 100]
