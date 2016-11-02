"""Description of this file."""

import os
import warnings

import numpy as np
import nibabel as ni
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
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

    # Extract the target ages from the data
    healthy = data["Y"].tolist()

    # print("Create young and old subsets")

    # Compute subsets of samples of young and old
    # young_inputs, young_ages = filter_samples(training_inputs, ages, 0, 60)
    # old_inputs, old_ages = filter_samples(training_inputs, ages, 60, 100)

    # print("Make a cross validated prediction for each age group")

    # Make a cross validated prediction for each age group
    predictor_all = cross_val_predict_data(training_inputs, healthy, "all")
    # predictor_young = cross_val_predict_data(young_inputs, young_ages, "young")
    # predictor_old = cross_val_predict_data(old_inputs, old_ages, "old")

    print("Load the test inputs")

    # Load the test inputs
    test_inputs = load_samples_inputs(False)
    test_inputs = pre_process(test_inputs)
    test_inputs = [binarize(i, 7) for i in test_inputs]

    print("Make an overall prediction for the test inputs")

    # Make an overall prediction for each test input
    test_predicted = predictor_all.predict(test_inputs)

    print("Refine predictions per age group")

    # Refine the prediction for each age group
    # for idx, p in enumerate(test_predicted):
    #     data_idx = test_inputs[idx]
    #     if p < 50:
    #         refined_prediction = predictor_young.predict(data_idx)
    #         test_predicted[idx] = refined_prediction
    #     if p >= 60:
    #         refined_prediction = predictor_old.predict(data_idx)
    #         test_predicted[idx] = refined_prediction

    # Make sure no prediction is below 18
    # test_predicted = [19 if p < 18 else p for p in test_predicted]

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


# def filter_samples(inputs, ages, age_min, age_max):
#     """
#     Create a subset of training samples for the given age range
#     :param inputs: List of inputs
#     :param ages: List of matching ages
#     :param age_min: Min age
#     :param age_max: Max age
#     :return: Lists of filtered inputs and ages
#     """
#
#     filtered_inputs = []
#     filtered_ages = []
#     for input_data, age in zip(inputs, ages):
#         if age_min < age <= age_max:
#             filtered_inputs.append(input_data)
#             filtered_ages.append(age)
#
#     return filtered_inputs, filtered_ages


def cross_val_predict_data(inputs, ages, tag="all"):
    """
    Make a cross validated prediction for the given inputs and ages
    :param inputs: The brain scans
    :param ages: The ages
    :param tag: Group tag
    :return: Sklearn predictor
    """

    # Make a histogram of 7 even bins for each brain scan
    bins = 7
    inputs = pre_process(inputs)
    inputs = [binarize(i, bins) for i in inputs]

    # Create the pipeline for a linear regression
    # on features of second order polynomials
    predictor = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearSVC(fit_intercept=False))
    ])

    # Cross validated prediction
    scores = cross_val_score(predictor, inputs, ages, scoring=make_scorer(log_loss), cv=4, n_jobs=4)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Output the MSE
    # print("{}: mse={}".format(
    #     tag,
    #     mean_squared_error(ages, predicted)
    # ))

    # Fit the predictor with the training data for later use
    predictor.fit(inputs, ages)

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
