"""Description of this file."""

import os
import warnings

import itertools

import collections

import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, hamming_loss
from sklearn.feature_selection import SelectKBest, chi2

from .load_data import load_samples_inputs, load_targets

from .settings import CURRENT_DIRECTORY

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
    bins_h = 12
    bins_a = 12
    bins_g = 8

    # Group statistics
    data["Y"] = 4 * data["H"] + 2 * data["A"] + data["G"]
    person = data["Y"].tolist()
    print(collections.Counter(person))

    # Extract the target ages from the data
    gender = data["G"].tolist()
    age = data["A"].tolist()
    health = data["H"].tolist()

    print("Make a cross validated prediction")
    predictor_gender = cross_val_score_data(training_inputs, gender, bins_g)
    predictor_age = cross_val_score_data(training_inputs, age, bins_a)
    predictor_health = cross_val_score_data(training_inputs, health, bins_h)

    print("Load the test inputs")

    # Load the test inputs
    test_inputs = load_samples_inputs(False)
    test_inputs_g = extract_features_hypercubes(test_inputs, bins_g)
    test_inputs_a = extract_features_hypercubes(test_inputs, bins_a)
    test_inputs_h = extract_features_hypercubes(test_inputs, bins_h)

    print("Make an overall prediction for the test inputs")

    # Make an overall prediction for each test input
    test_predicted_gender = predictor_gender.predict(test_inputs_g)
    test_predicted_age = np.array(predictor_age.predict(test_inputs_a))
    test_predicted_health = predictor_health.predict(test_inputs_h)

    print("Write prediction to predictions.csv")

    l = len(test_predicted_health)
    df = pd.DataFrame()
    df["ID"] = range(0, 3*l)
    df["Sample"] = list(itertools.chain(*[[i, i, i] for i in range(0, l)]))
    df["Label"] = list(itertools.chain(*[["gender", "age", "health"] for _ in range(0, l)]))

    df["Predicted"] = list(itertools.chain(*[
        [t[0] > 0.5, t[1] > 0.5, t[2] > 0.5] for t in zip(
            test_predicted_gender,
            test_predicted_age,
            test_predicted_health
        )]))

    prediction_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "data",
        "predictions.csv"
    )
    df.to_csv(prediction_path, index=False)


def label_to_triplet(l):
    return [l % 2 == 1, (l-l % 2) % 4 == 2, l > 4]


class PredictorWrapper:
    def __init__(self, predictorInstance):
        self.predictor = predictorInstance

    def fit(self, *args, **kwargs):
        self.predictor.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        test_predicted = self.predictor.predict_proba(*args, **kwargs)
        test_predicted = [
            np.argmax(ps)
            for ps in test_predicted
        ]
        return test_predicted


def cross_val_score_data(inputs, labels, bins):
    """
    Make a cross validated prediction for the given inputs and ages
    :param inputs: The brain scans
    :param tag: Group tag
    :return: Sklearn predictor
    """

    print("Extract features")
    print(collections.Counter(labels))

    # Make a histogram of even bins for each brain scan
    inputs = extract_features_hypercubes(inputs, bins)

    # Create the pipeline for a linear regression
    # on features of second order polynomials
    predictor = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        # ('feature_selection', SelectKBest(chi2, k=30)),
        ('linear', PredictorWrapper(LogisticRegression()))
    ])

    print("Compute cross validated score")

    # Cross validated prediction
    scores = cross_val_score(
        predictor,
        inputs,
        labels,
        scoring=make_scorer(hamming_loss),
        cv=4,
        n_jobs=4
    )
    print("Log Loss: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

    # Fit the predictor with the training data for later use
    predictor.fit(inputs, labels)

    return predictor


def extract_features_hypercubes(data, bins):
    """
    Flatten the brain scans to a 1D numpy array
    :param data: List of 3D memory map
    :return: List of 1D numpy arrays
    """

    lx = len(data[0].get_data()[:, 0, 0, 0])
    ly = len(data[0].get_data()[0, :, 0, 0])
    lz = len(data[0].get_data()[0, 0, :, 0])

    x_intervals = generate_intervals(lx, 3)
    y_intervals = generate_intervals(ly, 3)
    z_intervals = generate_intervals(lz, 3)

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


def binarize(i, bins):
    """
    Create evenly spaced bins in the meaningful range
    Weight each bin by its average value
    :param i: Inputs
    :param bins: Bins
    :return: Weighted bin values
    """

    hist, edges = np.histogram(i, bins=bins, range=[10, 2000], normed=True)
    edges = (edges[:-1] + edges[1:])/2
    hist *= edges

    return hist
