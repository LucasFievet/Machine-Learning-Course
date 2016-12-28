"""This script creates the final submission uploaded to Kaggle."""

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

from .settings import CURRENT_DIRECTORY

warnings.filterwarnings("ignore", category=DeprecationWarning)


def submission_predictor():
    """
    Outputs the predictions for the test set into final_sub.csv
    :return: None
    """

    print("Load inputs")

    # Load the training inputs and targets
    training_inputs = load_samples_inputs()
    data = load_targets()
    bins = 12

    # Extract the target ages from the data
    health = data["Y"].tolist()

    print("Make a cross validated prediction")

    predictor_all = cross_val_predict_data(training_inputs, health, bins)
    cross_val_score_data(training_inputs, health, bins)

    print("Load the test inputs")

    # Load the test inputs
    test_inputs = load_samples_inputs(False)
    test_inputs = extract_features_regions(test_inputs, bins)

    print("Make an overall prediction for the test inputs")

    # Make an overall prediction for each test input
    test_predicted = predictor_all.predict(test_inputs)

    print("Write prediction to final_sub.csv")

    df = pd.DataFrame()
    df["ID"] = range(1, len(test_predicted) + 1)
    df["Prediction"] = test_predicted

    prediction_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "final_sub.csv"
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


def cross_val_score_data(inputs, labels, bins):
    """
    Make a cross validated prediction for the given inputs and ages
    :param inputs: The brain scans
    :param tag: Group tag
    :return: Sklearn predictor
    """

    print("Extract features")

    # Make a histogram of 12 even bins for each brain scan
    inputs = extract_features_regions(inputs, bins)

    # Create the pipeline for a logistic regression
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
        labels,
        scoring=make_scorer(log_loss),
        cv=4,
        n_jobs=4
    )
    print("Log Loss: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

    # Fit the predictor with the training data for later use
    predictor.fit(inputs, labels)

    return predictor


def cross_val_predict_data(inputs, labels, bins):
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
    predicted = cross_val_predict(
        predictor,
        inputs,
        labels,
        # scoring=make_scorer(log_loss),
        cv=4,
        n_jobs=4
    )
    loss = log_loss(labels, predicted)
    print(predicted)
    print(loss)
    # print("Log Loss: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

    # Fit the predictor with the training data for later use
    predictor.fit(inputs, labels)

    return predictor


def extract_features_regions(data, bins):
    """
    Flatten the brain scans to a 1D numpy array
    :param data: List of 3D memory map
    :return: List of 1D numpy arrays
    """

    inputs = []
    for i in data:
        flat_data_i = i.get_data().flatten()
        features = binarize(
            flat_data_i,
            bins
        )

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

    hist, edges = np.histogram(i, bins=bins, range=[10, 2000], normed=True)
    edges = (edges[:-1] + edges[1:])/2
    hist *= edges

    return hist


def generate_intervals(l, n):
    step = int(l/n)
    return [
        (i1, i2)
        for i1, i2 in zip(range(0, l, step), range(step, l+1, step))
    ]
