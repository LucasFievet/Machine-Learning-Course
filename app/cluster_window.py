import os
import sys

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from scipy.stats import linregress

import itertools

from .load_data import load_targets, load_samples_inputs

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso, Huber
from sklearn.cross_validation import cross_val_predict
from sklearn.decomposition import IncrementalPCA
from .histogram_plot import histogram_plot

from .settings import CURRENT_DIRECTORY

def get_window_age_correlating():
    cache_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "cache",
        "window-age-correlating-training.hdf"
    )

    if os.path.exists(cache_path):
        data = pd.read_hdf(cache_path, "table")
    else:
        data = window_age_correlation_compute()
        data.to_hdf(cache_path, "table")

    print("")
    print(data)

    return data

def window_age_correlation_compute():
    training_inputs = load_samples_inputs()
    data = load_targets()
    ages = data["Y"].tolist()

    inputs = [i.get_data() for i in training_inputs]

    w_size = 10
    window = np.zeros((w_size,w_size,w_size))
    w_range = range(0,w_size)

    l_x = len(inputs[0][:, 0, 0, 0])
    l_y = len(inputs[0][0, :, 0, 0])
    l_z = len(inputs[0][0, 0, :, 0])

    x_range = range(0, l_x-w_size)
    y_range = range(0, l_y-w_size)
    z_range = range(0, l_z-w_size)

    slopes = []
    rs = []
    ps = []
    count = 0

    rc = np.zeros(inputs[0][:,:,:,0].shape)

    for c in itertools.product(x_range, y_range, z_range):
        print_progress("({}, {}, {})".format(*c))

        vs = []
        for i in inputs:
            for w in itertools.product(w_range,w_range,w_range):
                window[w[0]][w[1]][w[2]] = i[c[0]+w[0], c[1]+w[1], c[2]+w[2], 0]
            w_data = window.flatten()
            vs.append(np.mean(w_data))

        slope, intercept, r, p, std = linregress(ages, vs)
        slopes.append(slope)
        rs.append(r**2)
        ps.append(p)

        if r**2 > 0.1:
            rc[c[0]][c[1]][c[2]] = r**2 

    histogram_plot(slopes, "slopes")
    histogram_plot(rs, "rs")
    histogram_plot(ps, "ps")


    return rc 


def print_progress(s):
    sys.stdout.write("\r%s" % s)
    sys.stdout.flush()
