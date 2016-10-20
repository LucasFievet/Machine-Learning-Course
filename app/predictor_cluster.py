"""Description of this file."""


import os

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from .feature1 import feature_mean, feature_ratio, feature_max
from .load_data import load_targets, load_samples_inputs

from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import cross_val_predict
from sklearn.decomposition import IncrementalPCA
from .histogram_plot import histogram_plot
from .heatmap import heatmap

from .settings import CURRENT_DIRECTORY


__author__ = "lfievet"
__copyright__ = "Copyright 2016, Project One"
__credits__ = ["lfievet"]
__license__ = "No License"
__version__ = "1.0"
__maintainer__ = "lfievet"
__email__ = "lfievet@ethz.ch"
__date__ = "19/10/2016"
__status__ = "Production"


def predict_cluster():
    cache_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "cache",
        "trial1.hdf"
    )
    # if os.path.exists(cache_path):
    #     data = pd.read_hdf(cache_path, "table")
    # else:
    #     data = load_features()
    #
    # data.to_hdf(cache_path, "table")

    get_brain_pca()


def get_brain_pca():
    inputs = load_samples_inputs()
    data = load_targets()
    histogram_plot(data["Y"].tolist(), "ages")
    return

    s = 0
    l = len(inputs)

    mean = divide(inputs[0], l)

    for i in range(s, s+l):
        v = divide(inputs[i], l)
        mean = np.add(mean, v)

    print(np.mean(mean))

    histogram_plot(mean.flatten(), "hist-mean")

    pca_l = 125
    pca_inputs = []
    for i in range(0, pca_l):
        pca_input = inputs[i].get_data()[:, :, :, 0] - mean[:, :, :, 0]
        pca_input = pca_input.flatten()
        pca_inputs.append(pca_input)

    pca = IncrementalPCA(n_components=5, batch_size=5)
    pca.fit(pca_inputs)
    # print(pca.components_)
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)
    # print(pca.mean_)
    # print(pca.var_)
    # print(pca.n_samples_seen_)

    pca_values = pca.transform(pca_inputs)
    print(pca_values)

    for i in range(0, 10):
        features = list(map(lambda x: x[i], pca_values))
        ages = data["Y"].tolist()[:pca_l]
        correlation_plot(features, ages, "correlation-{}".format(i))


def correlation_plot(features, ages, filename):
    print(features)
    print(ages)
    print(len(features))
    print(len(ages))

    plt.figure()
    plt.scatter(ages, features)
    plt.xlabel('Age')
    plt.ylabel('PCA-1')
    plt.title(r'$\mathrm{PCA-1 as function of age}$')
    plt.grid(True)
    plt.savefig("plots/{}.pdf".format(filename))


def divide(input, l):
    return np.divide(input.get_data(), l)
