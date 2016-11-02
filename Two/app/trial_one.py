"""Description of this file."""

import os
import warnings

import itertools

import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import log_loss, make_scorer

from .load_data import load_samples_inputs, load_targets

from .settings import CURRENT_DIRECTORY, DATA_DIRECTORY

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
    healthy = data["Y"].tolist()

    # Make a cross validated prediction for each age group
    predictor_all = cross_val_predict_data(training_inputs, healthy, bins)

    print("Load the test inputs")

    # Load the test inputs
    test_inputs = load_samples_inputs(False)
    test_inputs = pre_process(test_inputs, bins)
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
        test_predicted = [p[1] for p in test_predicted]
        test_predicted = [1.0 if p > 0.95 else p for p in test_predicted]
        return test_predicted


def cross_val_predict_data(inputs, healthy, bins):
    """
    Make a cross validated prediction for the given inputs and ages
    :param inputs: The brain scans
    :param healthy: The ages
    :param tag: Group tag
    :return: Sklearn predictor
    """

    # Make a histogram of 7 even bins for each brain scan
    inputs = pre_process(inputs, bins)
    # inputs = [binarize(i, bins) for i in inputs]

    # for k, a in itertools.product(range(10, 11, 1), ["tanh"]):
    #     print("%s: %s" % (k, a))
    # Create the pipeline for a linear regression
    # on features of second order polynomials
    predictor = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        # ('linear', PredictorWrapper(MLPClassifier(hidden_layer_sizes=(k, k, k, k), activation=a)))
        ('linear', PredictorWrapper(LogisticRegression()))
    ])

    # Cross validated prediction
    scores = cross_val_score(
        predictor,
        inputs,
        healthy,
        scoring=make_scorer(log_loss),
        cv=4,
        n_jobs=4
    )
    print("Log Loss: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Fit the predictor with the training data for later use
    predictor.fit(inputs, healthy)

    return predictor


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


def pre_process(data, bins):
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


def generate_intervals(l, n):
    step = int(l/n)
    return [
        (i1, i2)
        for i1, i2 in zip(range(0, l, step), range(step, l+1, step))
    ]
