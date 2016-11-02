"""Description of this file."""

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
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
    test_inputs = pre_process(test_inputs)
    test_inputs = [binarize(i, bins) for i in test_inputs]

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


class Predictor:
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1

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
        return [p[1] for p in test_predicted]


def cross_val_predict_data(inputs, healthy, bins):
    """
    Make a cross validated prediction for the given inputs and ages
    :param inputs: The brain scans
    :param healthy: The ages
    :param tag: Group tag
    :return: Sklearn predictor
    """

    print(healthy)
    p0 = float(len(list(filter(lambda x: x < 0.5, healthy))))/len(healthy)
    p1 = float(len(list(filter(lambda x: x > 0.5, healthy))))/len(healthy)
    print(p0)
    print(p1)
    # raise ""

    # Make a histogram of 7 even bins for each brain scan
    inputs = pre_process(inputs)
    inputs = [binarize(i, bins) for i in inputs]
    # inputs = range(0, 278)

    # Create the pipeline for a linear regression
    # on features of second order polynomials
    predictor = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', PredictorWrapper(LogisticRegression()))
    ])

    # Cross validated prediction
    scores = cross_val_score(predictor, inputs, healthy, scoring=make_scorer(log_loss), cv=4, n_jobs=4)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

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


def pre_process(data):
    """
    Flatten the brain scans to a 1D numpy array
    :param data: List of 3D memory map
    :return: List of 1D numpy arrays
    """

    return np.array([i.get_data()[:, :, :, 0].flatten() for i in data])
