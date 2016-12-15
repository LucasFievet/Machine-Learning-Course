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
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif


from .settings import CURRENT_DIRECTORY, CACHE_DIRECTORY
from .find_features import FindFeatures
from .trial_one import PredictorWrapper

from .load_data import load_targets
from .reduce_histogram import ReduceHistogram

warnings.filterwarnings("ignore", category=DeprecationWarning)

def predict(bin_size=200, box_size=20, thresh=20, fac=2):
    #features = FindFeatures(bin_size, box_size, thresh, fac)
    #train_data = features.get_auto_features(100)
    #targets = features.get_targets()
    targets = load_targets()['Y'].tolist()
    reduced = ReduceHistogram(150, 10)
    data = reduced.get_reduced_set('train')
    data = [data[i, :, :].flatten() for i in range(np.shape(data)[0])]

    data = SelectKBest(mutual_info_classif, k=1000).fit_transform(data, targets)
    scaler = preprocessing.Normalizer().fit(data)
    data = scaler.transform(data)

    predictor = cross_val_score_data(data, targets)


def cross_val_score_data(inputs, labels):
    """
    Make a cross validated prediction for the given inputs and ages
    :param tag: Group tag
    :return: Sklearn predictor
    """
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
        scoring=make_scorer(log_loss),
        cv=4,
        n_jobs=8
    )
    print("Log Loss: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

    # Fit the predictor with the training data for later use
    #predictor.fit(inputs, labels)

    return predictor
