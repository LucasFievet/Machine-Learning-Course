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
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.metrics import log_loss, make_scorer, hamming_loss

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

    # young = data[data["A"] == 1]
    # old = data[data["A"] == 0]
    # young_gender = young["G"].tolist()
    # old_gender = old["G"].tolist()

    # training_inputs_young = [training_inputs[i] for i in young.index.tolist()]
    # training_inputs_old = [training_inputs[i] for i in old.index.tolist()]

    # Make a cross validated prediction for each age group
    # predictor_gender_young = cross_val_score_data(training_inputs_young, young_gender, bins_g)
    # predictor_gender_old = cross_val_score_data(training_inputs_old, old_gender, bins_g)

    predictor_gender = cross_val_score_data(training_inputs, gender, bins_g)
    predictor_age = cross_val_score_data(training_inputs, age, bins_a)
    predictor_health = cross_val_score_data(training_inputs, health, bins_h)

    print("Load the test inputs")

    # Load the test inputs
    test_inputs = load_samples_inputs(False)
    test_inputs_g = extract_features_regions(test_inputs, bins_g)
    test_inputs_a = extract_features_regions(test_inputs, bins_a)
    test_inputs_h = extract_features_regions(test_inputs, bins_h)

    print("Make an overall prediction for the test inputs")

    # Make an overall prediction for each test input
    test_predicted_age = np.array(predictor_age.predict(test_inputs_a))
    # print(test_predicted_age)
    # young_indices = np.where(test_predicted_age > 0.5)[0]
    # old_indices = np.where(test_predicted_age <= 0)[0]
    # print(young_indices)
    # print(old_indices)
    # print(test_inputs)
    #
    # test_inputs_young = np.array([test_inputs[i] for i in young_indices])
    # test_inputs_old = np.array([test_inputs[i] for i in old_indices])

    # test_predicted_gender_young = predictor_gender_young.predict(test_inputs_g)
    # test_predicted_gender_old = predictor_gender_old.predict(test_inputs_g)
    #
    # test_predicted_gender = [0 for _ in range(0, len(test_predicted_age))]
    # for i in young_indices:
    #     test_predicted_gender[i] = test_predicted_gender_young[i]
    # for i in old_indices:
    #     test_predicted_gender[i] = test_predicted_gender_old[i]

    test_predicted_gender = predictor_gender.predict(test_inputs_g)
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
        labels,
        scoring=make_scorer(hamming_loss),
        cv=4,
        n_jobs=4
    )
    # print(scores)
    print("Log Loss: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

    # Fit the predictor with the training data for later use
    predictor.fit(inputs, labels)

    return predictor


# def cross_val_predict_data(inputs, labels, health, bins):
#     """
#     Make a cross validated prediction for the given inputs and ages
#     :param inputs: The brain scans
#     :param health: The ages
#     :param tag: Group tag
#     :return: Sklearn predictor
#     """
#
#     print("Extract features")
#
#     # Make a histogram of 7 even bins for each brain scan
#     inputs = extract_features_regions(inputs, bins)
#
#     # Create the pipeline for a linear regression
#     # on features of second order polynomials
#     predictor = Pipeline([
#         ('poly', PolynomialFeatures(degree=2)),
#         ('linear', PredictorWrapper2(LogisticRegression()))
#     ])
#
#     print("Compute cross validated score")
#
#     # Cross validated prediction
#     predicted = cross_val_predict(
#         predictor,
#         inputs,
#         labels,
#         # scoring=make_scorer(log_loss),
#         cv=4,
#         n_jobs=4
#     )
#     loss = log_loss(health, predicted)
#     print(predicted)
#     print(loss)
#     # print("Log Loss: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
#
#     # Fit the predictor with the training data for later use
#     predictor.fit(inputs, labels)
#
#     return predictor
#
#
# def cross_val_label(inputs, health, bins):
#     """
#     Make a cross validated prediction for the given inputs and ages
#     :param inputs: The brain scans
#     :param health: The ages
#     :param bins: Number bins
#     :return: Sklearn predictor
#     """
#
#     print("Extract features")
#
#     # Make a histogram of 7 even bins for each brain scan
#     inputs = extract_features_regions(inputs, bins)
#
#     # Create the pipeline for a linear regression
#     # on features of second order polynomials
#     predictor = Pipeline([
#         ('poly', PolynomialFeatures(degree=2)),
#         ('linear', PredictorWrapper(LogisticRegression()))
#     ])
#
#     print("Compute cross validate predict")
#
#     r = cross_val_predict(
#         predictor,
#         inputs,
#         health,
#         cv=4,
#         n_jobs=4
#     ) - health
#
#     import matplotlib.pyplot as plt
#     plt.figure()
#     plt.hist(
#         r,
#         bins=20,
#         normed=False,
#         facecolor='green',
#         alpha=0.75
#     )
#
#     plt.xlabel('Error')
#     plt.ylabel('Count')
#     plt.title(r'$\mathrm{Prediction Error}$')
#     plt.grid(True)
#     plt.savefig("plots/{}.pdf".format("hist"))
#
#     labels = []
#     for h, p in zip(health, r):
#         if h == 0 and p > 0.25:
#             labels.append(-1)
#         elif h == 1 and p < -0.25:
#             labels.append(2)
#         else:
#             labels.append(h)
#
#     return labels
#
#
# def extract_features_hypercubes(data, bins):
#     """
#     Flatten the brain scans to a 1D numpy array
#     :param data: List of 3D memory map
#     :return: List of 1D numpy arrays
#     """
#
#     lx = len(data[0].get_data()[:, 0, 0, 0])
#     ly = len(data[0].get_data()[0, :, 0, 0])
#     lz = len(data[0].get_data()[0, 0, :, 0])
#
#     x_intervals = generate_intervals(lx, 2)
#     y_intervals = generate_intervals(ly, 1)
#     z_intervals = generate_intervals(lz, 1)
#
#     print(x_intervals)
#     print(y_intervals)
#     print(z_intervals)
#
#     inputs = []
#     for i in data:
#         features = np.array([])
#         for ix, iy, iz in itertools.product(x_intervals, y_intervals, z_intervals):
#             features = np.concatenate((features, binarize(
#                 i.get_data()[ix[0]:ix[1], iy[0]:iy[1], iz[0]:iz[1], 0].flatten(),
#                 bins
#             )))
#
#         inputs.append(features)
#
#     return np.array(inputs)


def extract_features_regions(data, bins):
    """
    Flatten the brain scans to a 1D numpy array
    :param data: List of 3D memory map
    :return: List of 1D numpy arrays
    """

    # region_indices = []
    # region_path = os.path.join(
    #     CACHE_DIRECTORY,
    #     "region_cache.hdf"
    # )
    # for l in range(2, 3):
    #     df = pd.read_hdf(region_path.replace(".hdf", "-%s.hdf" % l), "table")
    #     region_indices.append(df[l].tolist())
    #     print(len(df))

    inputs = []
    for i in data:
        flat_data_i = i.get_data().flatten()
        features = binarize(
            flat_data_i,
            bins
        )

        # features = np.array([])
        # for r in region_indices:
        #     features = np.concatenate((features, binarize(
        #         flat_data_i,
        #         bins
        #     )))

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


# def generate_intervals(l, n):
#     step = int(l/n)
#     return [
#         (i1, i2)
#         for i1, i2 in zip(range(0, l, step), range(step, l+1, step))
#     ]
