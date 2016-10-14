"""Description of this file."""


import os

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from .feature_3d import feature_mean, feature_ratio, feature_max, feature_ratio_mean
from .load_data_3d import load_targets, load_samples_inputs
from .squared_error import squared_error

from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import accuracy_score

from .settings import CURRENT_DIRECTORY
from .cut_frontal_lobe import cut_frontal_lobe

def predict_test(training=True):
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
    xs = data[["mean", "ratio_mean", "max"]].values.tolist()
    ys = data["Y"].values.tolist()
    nn = KNeighborsRegressor(
        n_neighbors=4,
        weights="uniform",
        p=2,
        n_jobs=-1,
    )
    predicted = cross_val_predict(nn, xs, ys, cv=5)
    print(squared_error(ys,predicted))


def load_features():
    inputs = load_samples_inputs()
    data = load_targets()

    features = [
        {
            "name": "mean",
            "f": feature_mean
        },
        {
            "name": "ratio_mean",
            "f": feature_ratio_mean
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
