"""Description of this file."""

import os
import warnings

import itertools

import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.metrics import log_loss, make_scorer

from .load_data import load_samples_inputs, load_targets

from .settings import CURRENT_DIRECTORY, CACHE_DIRECTORY

warnings.filterwarnings("ignore", category=DeprecationWarning)


__author__ = "lfievet"
__copyright__ = "Copyright 2016, Project One"
__credits__ = ["lfievet"]
__license__ = "No License"
__version__ = "1.0"
__maintainer__ = "lfievet"
__email__ = "lfievet@ethz.ch"
__date__ = "25/10/2016"
__status__ = "Production"


def submission_predictor():
    """
    Outputs the predictions for the test set into predictions.csv
    :return: None
    """

    print("Load inputs")

    # Load the training inputs and targets
    training_inputs = load_samples_inputs()
    data = load_targets()
    bins = 10

    # Extract the target ages from the data
    health = data["Y"].tolist()

    print("Create better labels")

    labels = cross_val_label(training_inputs, health, bins)

    print("Make a cross validated prediction")

    # Make a cross validated prediction for each age group
    predictor_all = cross_val_predict_data(training_inputs, labels, bins)

    print("Load the test inputs")

    # Load the test inputs
    test_inputs = load_samples_inputs(False)
    test_inputs = extract_features_regions(test_inputs, bins)
    # test_inputs = [binarize(i, bins) for i in test_inputs]

    print("Make an overall prediction for the test inputs")

    # Make an overall prediction for each test input
    test_predicted = predictor_all.predict(test_inputs)

    print("Write prediction to predictions.csv")

    df = pd.DataFrame()
    df["ID"] = range(1, len(test_predicted) + 1)
    df["Prediction"] = test_predicted

    prediction_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "data",
        "predictions.csv"
    )
    df.to_csv(prediction_path, index=False)


class PredictorMeanProbability:
    def __init__(self):
        pass

    def fit(self, x, y):
        pass

    def predict(self, x):
        return [0.75 for _ in range(0, len(x))]


class PredictorWrapper:
    def __init__(self, predictorInstance):
        self.predictor = predictorInstance

    def fit(self, *args, **kwargs):
        self.predictor.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        test_predicted = self.predictor.predict_proba(*args, **kwargs)
        test_predicted = [
            p[1] if len(p) == 2 else p[2] + p[3]
            for p in test_predicted
        ]
        test_predicted = [1.0 if p > 0.95 else p for p in test_predicted]
        return test_predicted


def cross_val_predict_data(inputs, health, bins):
    """
    Make a cross validated prediction for the given inputs and ages
    :param inputs: The brain scans
    :param health: The ages
    :param tag: Group tag
    :return: Sklearn predictor
    """

    print("Extract features")

    # Make a histogram of 7 even bins for each brain scan
    inputs = extract_features_regions(inputs, bins)

    # Create the pipeline for a linear regression
    # on features of second order polynomials
    predictor = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', PredictorWrapper(LogisticRegression()))
    ])

    print("Compute cross validated score")

    # Cross validated prediction
    scores = cross_val_score(
        predictor,
        inputs,
        health,
        scoring=make_scorer(log_loss),
        cv=4,
        n_jobs=4
    )
    print("Log Loss: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Fit the predictor with the training data for later use
    predictor.fit(inputs, health)

    return predictor


def cross_val_label(inputs, health, bins):
    """
    Make a cross validated prediction for the given inputs and ages
    :param inputs: The brain scans
    :param health: The ages
    :param bins: Number bins
    :return: Sklearn predictor
    """

    print("Extract features")

    # Make a histogram of 7 even bins for each brain scan
    inputs = extract_features_regions(inputs, bins)

    # Create the pipeline for a linear regression
    # on features of second order polynomials
    predictor = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', PredictorWrapper(LogisticRegression()))
    ])

    print("Compute cross validate predict")

    r = cross_val_predict(
        predictor,
        inputs,
        health,
        cv=4,
        n_jobs=4
    ) - health

    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(
        r,
        bins=20,
        normed=False,
        facecolor='green',
        alpha=0.75
    )

    plt.xlabel('Error')
    plt.ylabel('Count')
    plt.title(r'$\mathrm{Prediction Error}$')
    plt.grid(True)
    plt.savefig("plots/{}.pdf".format("hist"))

    labels = []
    for h, p in zip(health, r):
        if h == 0 and p > 0.25:
            labels.append(-1)
        elif h == 1 and p < -0.25:
            labels.append(2)
        else:
            labels.append(h)

    return labels


def extract_features_hypercubes(data, bins):
    """
    Flatten the brain scans to a 1D numpy array
    :param data: List of 3D memory map
    :return: List of 1D numpy arrays
    """

    lx = len(data[0].get_data()[:, 0, 0, 0])
    ly = len(data[0].get_data()[0, :, 0, 0])
    lz = len(data[0].get_data()[0, 0, :, 0])

    x_intervals = generate_intervals(lx, 2)
    y_intervals = generate_intervals(ly, 1)
    z_intervals = generate_intervals(lz, 1)

    print(x_intervals)
    print(y_intervals)
    print(z_intervals)

    inputs = []
    for i in data:
        features = np.array([])
        for ix, iy, iz in itertools.product(x_intervals, y_intervals, z_intervals):
            features = np.concatenate((features, binarize(
                i.get_data()[ix[0]:ix[1], iy[0]:iy[1], iz[0]:iz[1], 0].flatten(),
                bins
            )))

        inputs.append(features)

    return np.array(inputs)


def extract_features_regions(data, bins):
    """
    Flatten the brain scans to a 1D numpy array
    :param data: List of 3D memory map
    :return: List of 1D numpy arrays
    """

    region_indices = []
    region_path = os.path.join(
        CACHE_DIRECTORY,
        "region_cache.hdf"
    )
    for l in range(0, 1):
        df = pd.read_hdf(region_path.replace(".hdf", "-%s.hdf" % l), "table")
        region_indices.append(df[l].tolist())
        print(len(df))

    inputs = []
    for i in data:
        flat_data_i = i.get_data().flatten()

        features = np.array([])
        for r in region_indices:
            features = np.concatenate((features, binarize(
                flat_data_i,
                bins
            )))

        inputs.append(features)

    return np.array(inputs)


def binarize(i, bins):
    """
    Create evenly spaced bins in the meaningful range
    Weight each bin by its average value
    :param i: Inputs
    :param bins: Bins
    :return: Weighted bin values
    """

    hist, edges = np.histogram(i, bins=bins, range=[100, 1600], normed=True)
    edges = (edges[:-1] + edges[1:])/2
    hist *= edges

    return hist


def generate_intervals(l, n):
    step = int(l/n)
    return [
        (i1, i2)
        for i1, i2 in zip(range(0, l, step), range(step, l+1, step))
    ]
