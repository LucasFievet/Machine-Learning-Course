"""Description of this file."""


import os

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from .feature1 import feature_mean, feature_ratio, feature_max
from .load_data import load_targets, load_samples_inputs

from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import accuracy_score

from .settings import CURRENT_DIRECTORY


__author__ = "lfievet"
__copyright__ = "Copyright 2016, Project One"
__credits__ = ["lfievet"]
__license__ = "No License"
__version__ = "1.0"
__maintainer__ = "lfievet"
__email__ = "lfievet@ethz.ch"
__date__ = "12/10/2016"
__status__ = "Production"


def predict(training=True):
    cache_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "cache",
        "trial1.hdf"
    )
    if os.path.exists(cache_path):
        data = pd.read_hdf(cache_path, "table")
    else:
        data = load_features()

    data.to_hdf(cache_path, "table")

    print(data)
    xs = data[["mean", "max"]].values.tolist()
    ys = data["Y"].values.tolist()

    nn = KNeighborsRegressor(
        n_neighbors=4,
        weights="uniform",
        p=2,
    )
    predicted = cross_val_predict(nn, xs, ys, cv=5)
    diffs = ys - predicted
    print(np.mean(abs(diffs)))


def load_features():
    inputs = load_samples_inputs()
    data = load_targets()

    features = [
        {
            "name": "mean",
            "f": feature_mean
        },
        {
            "name": "ratio",
            "f": feature_ratio
        },
        {
            "name": "max",
            "f": feature_max
        },
    ]

    for f in features:
        feature_inputs = f["f"](inputs)
        data[f["name"]] = feature_inputs

        plt.figure()
        plt.scatter(
            feature_inputs,
            data["Y"].tolist(),
        )
        plt.savefig("plots/line_{}.pdf".format(
            f["name"]
        ))

    return data
