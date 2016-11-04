"""Description of this file."""

import os

import pandas as pd
import numpy as np

from functools import reduce

from .load_data import load_samples_inputs, load_targets

from sklearn.cluster import KMeans

from .settings import PLOT_DIRECTORY, CACHE_DIRECTORY


__author__ = "lfievet"
__copyright__ = "Copyright 2016, Project One"
__credits__ = ["lfievet"]
__license__ = "No License"
__version__ = "1.0"
__maintainer__ = "lfievet"
__email__ = "lfievet@ethz.ch"
__date__ = "02/11/2016"
__status__ = "Production"


def healthy_sick_comparison():
    """
    Outputs the predictions for the test set into predictions.csv
    :return: None
    """

    print("Load inputs")

    # Load the training inputs and targets
    training_inputs = load_samples_inputs()
    targets = load_targets()["Y"].tolist()

    healthy = get_class(training_inputs, targets, 1)
    sick = get_class(training_inputs, targets, 0)

    means, stds = get_stds(healthy)

    print("Sick")
    indices = []
    for idx, s in enumerate(sick):
        indices.append(get_sick_indices(means, stds, s))

    corr = np.zeros((len(indices), len(indices)))
    for i in range(0, len(indices)):
        for j in range(0, len(indices)):
            idx = indices[i]
            jdx = indices[j]
            corr[i, j] = np.intersect1d(idx, jdx).size/max(idx.size, jdx.size)

    cluster = KMeans(n_clusters=4)
    cluster.fit(corr)
    print(cluster.labels_)

    sorted_sick = np.array([], dtype=int)
    for l in range(0, 4):
        sorted_sick = np.concatenate((sorted_sick, np.where(np.array(cluster.labels_) == l)[0]))

    for i in range(0, len(indices)):
        for j in range(0, len(indices)):
            ix = sorted_sick[i]
            jx = sorted_sick[j]
            idx = indices[ix]
            jdx = indices[jx]
            corr[i, j] = np.intersect1d(idx, jdx).size/max(idx.size, jdx.size)

    import matplotlib.pyplot as plt

    plt.clf()
    plt.imshow(corr, origin='lower')

    plot_path = os.path.join(
            PLOT_DIRECTORY,
            "heatmap.pdf"
            )
    plt.savefig(plot_path)

    region_path = os.path.join(
            CACHE_DIRECTORY,
            "region_cache.hdf"
            )
    for l in range(0, 4):
        l_sick = np.array(indices)[np.array(cluster.labels_) == l]
        l_indices = reduce(np.union1d, l_sick)
        df = pd.DataFrame()
        df[l] = l_indices
        df.to_hdf(region_path.replace(".hdf", "-%s.hdf" % l), "table")

    # print("Healthy")
    # for s in healthy:
    #     get_sick_indices(means, stds, s)


def get_sick_indices(means, stds, sick):
    t = 4

    sick = sick.get_data()[:, :, :, 0].flatten()
    indices_up = np.logical_and(means + t*stds < sick, sick <= 1600)
    indices_down = np.logical_and(sick < means - t*stds, sick >= 100)
    print("{}: {}/{}".format(
            6,
            np.sum(indices_up),
            np.sum(indices_down)
            ))

    indices = np.logical_or(indices_up, indices_down)
    return np.where(indices)[0]


def get_stds(inputs):
    inputs = [
            np.minimum(np.array(i.get_data()[:, :, :, 0].flatten()), 1600)
            for i in inputs
            ]

    means = np.mean(inputs, axis=0)
    stds = np.std(inputs, axis=0)

    return means, stds


def get_class(inputs, targets, target):
    class_inputs = []
    for i, t in zip(inputs, targets):
        if t == target:
            class_inputs.append(i)

    return class_inputs
