import os
import sys

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from scipy.stats import linregress
import scipy.ndimage.filters as filters
import scipy.io

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

def cluster_mean(data):
    return [[np.mean(k.flatten()) for k in d] for d in data]

def get_clusters(w_size=10, thresh=0):
    cache_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "cache",
        "cluster_areas_training.hdf",
    )
    if os.path.exists(cache_path):
        data = pd.read_hdf(cache_path, "table")
    else:
        data = get_cluster_areas(w_size, thresh)
        data.to_hdf(cache_path, "table")

    return data

def get_cluster_areas(w_size=10, thresh=0):
    inputs = load_samples_inputs()
    correlation = get_window_age_correlating(w_size)
    window = combute_bool_cluster(correlation,w_size,thresh) 
    data = [compute_cluster_areas(i.get_data(),window,w_size) for i in inputs]
    return data

def combute_bool_cluster(data, w_size=10, thresh=0):
    l = local_max_loactions(data, w_size, thresh)
    out = np.zeros(data.shape)
    nol = len(l[0])
    print("maximas found:",nol)
    for i in range(nol):
        out[l[0][i],l[1][i],l[2][i]] = 1 
    return out 

def local_max_loactions(data, w_size, thresh=0):
    out = []
    shape = data.shape
    f = filters.maximum_filter(data,size=w_size) 
    for c in itertools.product(range(shape[0]),range(shape[1]),range(shape[2])):
        if f[c[0],c[1],c[2]] == data[c[0],c[1],c[2]]:
            if f[c[0],c[1],c[2]] > thresh:
                out.append(c)
    print(len(out))
    return out 

def compute_cluster_areas(data, window, w_size=10):
    out = []
    shape = window.shape
    w_range = range(w_size)
    for c in itertools.product(range(shape[0]-w_size),range(shape[1]-w_size),range(shape[2]-w_size)):
        print_progress("({}, {}, {})".format(*c))
        if window[c[0]][c[1]][c[2]] == 1:
            tmp = np.zeros(w_size,w_size,w_size)
            for w in itertools.product(w_range,w_range,w_range):
                tmp[w[0]][w[1]][w[2]] = data[c[0]+w[0], c[1]+w[1], c[2]+w[2], 0]
            out.append(tmp)
    return out

def get_window_age_correlating(w_size=10):
    cache_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "cache",
        "window-age-correlating-training.mat"
    )

    if os.path.exists(cache_path):
        data = scipy.io.loadmat(cache_path)['out']
    else:
        data = window_age_correlation_compute(w_size)
        scipy.io.savemat(cache_path, mdict={'out': data}, oned_as='row')

    print("")
    print(data)

    return data

def window_age_correlation_compute(w_size=10):
    training_inputs = load_samples_inputs()
    data = load_targets()
    ages = data["Y"].tolist()

    inputs = [i.get_data() for i in training_inputs]

    window = np.zeros((w_size,w_size,w_size))
    w_range = range(0,w_size)

    steps = 2

    l_x = len(inputs[0][:, 0, 0, 0])-w_size
    l_y = len(inputs[0][0, :, 0, 0])-w_size
    l_z = len(inputs[0][0, 0, :, 0])-w_size

    x_range = range(0, l_x, steps)
    y_range = range(0, l_y, steps)
    z_range = range(0, l_z, steps)

    slopes = []
    rs = []
    ps = []
    rc = np.zeros(inputs[0][:,:,:,0].shape)

    for c in itertools.product(x_range, y_range, z_range):
        print_progress("({}, {}, {})({},{},{})".format(c[0],c[1],c[2],l_x,l_y,l_z))

        vs = []
        for i in inputs:
            mean = 0
            for w in itertools.product(w_range,w_range,w_range):
                mean += i[c[0]+w[0], c[1]+w[1], c[2]+w[2], 0]
            mean /= w_size**3
            vs.append(mean)

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
