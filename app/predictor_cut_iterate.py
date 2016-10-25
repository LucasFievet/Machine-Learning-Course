"""Description of this file."""

import os

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import accuracy_score

from .settings import CURRENT_DIRECTORY, ITERATE_DIRECTORY
from .cut_brain import cut_brain
from .feature import feature_mean, feature_max, feature_ratio_mean
from .load_data_3d import load_targets, load_samples_inputs
from .squared_error import squared_error
from .subset_generator import all_subsets

def predict_cut_iterate(num, training=True):
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


    #print(data)
    feature_list = data.keys().tolist()
    feature_list.remove("Y")
    feature_subsets = all_subsets(feature_list,3)
    ys = data["Y"].values.tolist()
    nn = KNeighborsRegressor(
            n_neighbors=3,
            weights="uniform",
            p=2,
            n_jobs=-1,
            )

    DIR = ITERATE_DIRECTORY
    #files = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    steps = 10000
    start = num * steps
    end = start + steps
    if end > len(feature_subsets):
        end = len(feature_subsets)
    se = []
    print(len(feature_subsets))
    print("Starting Predictions")
    for subset in feature_subsets[start:end]:
        xs = data[subset].values.tolist()
        predicted = cross_val_predict(nn, xs, ys, cv=5)
        se.append([squared_error(ys,predicted),subset])

    se.sort(key=lambda x: x[0])
    print(se[0])
    iterate_path = os.path.join(ITERATE_DIRECTORY,"se_{}.hdf".format(num))
    pd.Series(se[0:10]).to_hdf(iterate_path, "table")


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
