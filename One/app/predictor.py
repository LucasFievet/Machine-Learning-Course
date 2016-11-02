"""Description of this file."""


import os

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from .feature1 import feature_mean, feature_ratio, feature_max
from .load_data import load_targets, load_samples_inputs

from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import cross_val_predict
from sklearn.linear_model import Perceptron

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

    # data = data[data["max"] > 600]
    # data = data[data["max"] < 1000]

    # data = data[data["max"] > 0]

    print(data)

    min = 1000
    for i in range(0, 6):
        for j in range(i+1, 6):
            for k in range(1, 10):
                xs = data[[i, j]].values.tolist()
                ys = data["Y"].values.tolist()

                nn = KNeighborsRegressor(
                    n_neighbors=k,
                    weights="uniform",
                    p=2,
                )
                # model = Perceptron()
                predicted = cross_val_predict(nn, xs, ys, cv=5)
                diffs = predicted - ys
                # print(diffs)
                mean_squared_error = np.mean(list(map(lambda x: x*x, diffs)))

                if mean_squared_error > min:
                    continue

                min = mean_squared_error
                print("-"*50)
                print(i)
                print(j)
                print(mean_squared_error)
                print(np.std(diffs))

                plt.figure()
                plt.scatter(
                    ys,
                    diffs,
                )
                plt.savefig("plots/diffs.pdf")


def load_features():
    inputs = load_samples_inputs()
    data = load_targets()

    features = [
        # {
        #     "name": "mean",
        #     "f": feature_mean
        # },
        # {
        #     "name": "ratio",
        #     "f": feature_ratio
        # },
        {
            "name": "max",
            "f": feature_max
        },
    ]

    for f in features:
        feature_inputs = f["f"](inputs)
        # data[f["name"]] = feature_inputs
        data = pd.concat([data, feature_inputs], axis=1)
        data.dropna(inplace=True)
        print(data)

        for i in range(0, 6):
            plt.figure()
            df = data[abs(data[i]) > 0]
            xs = df[i].tolist()
            plt.scatter(
                xs,
                df["Y"].tolist(),

            )
            # plt.xlim(-min(xs),  max(xs))
            plt.savefig("plots/line-{}-{}.pdf".format(
                f["name"],
                i
            ))

    return data
