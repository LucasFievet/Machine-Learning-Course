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
from .cut_brain import cut_brain

def predict_cut(training=True):
    cache_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "cache",
        "trial1.hdf"
    )
    if os.path.exists(cache_path):
        print("Loading features from cache")
        data = pd.read_hdf(cache_path, "table")
    else:
        print("Loading features")
        data = load_features()

    print("saving data to cache")
    data.to_hdf(cache_path, "table")

    print(data)
    feature_list = data.keys().tolist()
    feature_list.remove("Y")
    #feature_list = ["max_lb", "max_mb", "mean_mb", "mean_rt", "mean_mt", "ratio_mean_lt",
    #        "ratio_mean_mb", "ratio_mean_mt", "ratio_mean_rt", "ratio_mean_whole"]
    xs = data[feature_list].values.tolist()
    ys = data["Y"].values.tolist()
    nn = KNeighborsRegressor(
        n_neighbors=3,
        weights="uniform",
        p=2,
        n_jobs=-1,
    )
    print("Starting Prediction")
    predicted = cross_val_predict(nn, xs, ys, cv=5)
    print("Squared Error:",squared_error(ys,predicted))


def load_features():
    areas = ["lt","mt","rt","lb","mb","rb"]
    inputs = [
        {
            "area": "whole",
            "val": load_samples_inputs()
        }
    ]
    for a in areas:
        inputs.append(
            {
                "area": a,
                "val": cut_brain(inputs[0]["val"], a)
            }
            )
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

    print("plotting features")
    for f in features:
        for i in inputs:
            feature_inputs = f["f"](i["val"])
            data["{}_{}".format(f["name"], i["area"])] = feature_inputs

            plt.figure()
            plt.scatter(
                feature_inputs,
                data["Y"].tolist(),
            )
            plt.savefig("plots/line_{}_{}.pdf".format(
                f["name"], i["area"]
            ))
            plt.close()

    return data
