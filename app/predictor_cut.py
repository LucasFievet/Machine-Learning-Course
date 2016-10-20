"""Description of this file."""

import os

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import accuracy_score

from .settings import CURRENT_DIRECTORY
from .squared_error import squared_error
from .load_features import load_features

def predict_cut(training=True):
    cache_data_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "cache",
        "data.hdf"
    )
    cache_norms_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "cache",
        "norms.hdf"
    )
    cache_test_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "cache",
        "test.hdf"
    )
    if os.path.exists(cache_data_path) and os.path.exists(cache_norms_path):
        print("Loading features from cache")
        data = pd.read_hdf(cache_data_path, "table")
        norms = pd.read_hdf(cache_norms_path, "table").to_dict()
    else:
        print("Loading features")
        data, norms= load_features()
        print("saving data to cache")
        data.to_hdf(cache_data_path, "table")
        pd.DataFrame.from_dict(norms).to_hdf(cache_norms_path, "table")

    if os.path.exists(cache_test_path):
        print("Loading test features from cache")
        test_data = pd.read_hdf(cache_test_path, "table")
    else:
        print("Loading test features")
        test_data = load_features(norms)
        print("saving test to cache")
        test_data.to_hdf(cache_data_path, "table")

    feature_list = ['mean_rt', 'mean_mb', 'ratio_mean_lt', 'ratio_mean_rt', 'ratio_mean_rb', 'max_rt', 'max_rb']
    xs = data[feature_list].values.tolist()
    ys = data["Y"].values.tolist()
    test_input = test_data[feature_list].values.tolist()
    nn = KNeighborsRegressor(
        n_neighbors=3,
        weights="uniform",
        p=2,
        n_jobs=-1,
    )
    print("Starting Prediction")
    predicted = cross_val_predict(nn, xs, ys, cv=5)
    print("Squared Error:",squared_error(ys,predicted))

    nn.fit(xs, ys)

    result = nn.predict(test_input)
    print(result)
    result_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "result.csv"
    )
    result.to_csv(result_path)

