"""Description of this file."""

import os
import sys

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from .load_data import load_targets, load_samples_inputs

from sklearn.linear_model import Lasso
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import mean_squared_error

from scipy.stats import linregress

from .settings import CURRENT_DIRECTORY


__author__ = "lfievet"
__copyright__ = "Copyright 2016, Project One"
__credits__ = ["lfievet"]
__license__ = "No License"
__version__ = "1.0"
__maintainer__ = "lfievet"
__email__ = "lfievet@ethz.ch"
__date__ = "25/10/2016"
__status__ = "Production"


def scan_volume():
    training_inputs = load_samples_inputs()
    data = load_targets()
    ages = data["Y"].tolist()

    young_inputs, young_ages = filter_samples(training_inputs, ages, 0, 60)
    old_inputs, old_ages = filter_samples(training_inputs, ages, 60, 100)

    predictor_all = cross_val_predict_data(training_inputs, ages, "all")
    predictor_young = cross_val_predict_data(young_inputs, young_ages, "young")
    predictor_old = cross_val_predict_data(old_inputs, old_ages, "old")

    test_inputs = load_samples_inputs(False)
    test_data = [i.get_data()[:, :, :, 0].flatten() for i in test_inputs]
    test_data = [np.histogram(i, bins=30, range=[100, 1600])[0] for i in test_data]

    ratios1_test = get_ratios(test_data, predictor_all.best_p)
    ratios2_test = get_ratios(test_data, predictor_all.best_n)
    ratios_test = [[r1, r2] for r1, r2 in zip(ratios1_test, ratios2_test)]

    test_predicted = predictor_all.predict(ratios_test)

    for idx, p in enumerate(test_predicted):
        data_idx = test_data[idx]
        if p < 50:
            refined_prediction = predict_sample(predictor_young, data_idx)
            test_predicted[idx] = refined_prediction
            print("{}: {}".format(p, refined_prediction))
        if p >= 65:
            refined_prediction = predict_sample(predictor_old, data_idx)
            test_predicted[idx] = refined_prediction
            print("{}: {}".format(p, refined_prediction))

    test_predicted = [19 if p < 18 else p for p in test_predicted]
    print(test_predicted)
    print(min(test_predicted))
    print(max(test_predicted))
    df = pd.DataFrame()
    df["ID"] = range(1, len(test_predicted) + 1)
    df["Prediction"] = test_predicted

    prediction_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "data",
        "predictions-3.csv"
    )
    df.to_csv(prediction_path, index=False)


def predict_sample(predictor, data):
    r1 = get_ratio(data, predictor.best_p)
    r2 = get_ratio(data, predictor.best_n)

    return predictor.predict([r1, r2])[0]


def filter_samples(inputs, ages, age_min, age_max):
    filtered_inputs = []
    filtered_ages = []
    for input_data, age in zip(inputs, ages):
        if age_min < age < age_max:
            filtered_inputs.append(input_data)
            filtered_ages.append(age)

    return filtered_inputs, filtered_ages


def cross_val_predict_data(inputs, ages, tag="all"):
    bins = 30
    scan = True

    if tag == "all":
        scan = False
        best_p_xs = 8, 9, 10, 11
        best_n_xs = 12, 14, 15, 23
    elif tag == "young":
        scan = False
        best_p_xs = 0, 5, 24, 29
        best_n_xs = 12, 14, 16, 27
    elif tag == "old":
        scan = False
        best_p_xs = 2, 5, 24, 29
        best_n_xs = 13, 14, 18, 23
    else:
        best_p_xs = 0, 0, 0, 0
        best_n_xs = 0, 0, 0, 0

    inputs = [i.get_data()[:, :, :, 0].flatten() for i in inputs]
    inputs = [np.histogram(i, bins=bins, range=[100, 1600])[0] for i in inputs]
    best_p_r = 0
    best_n_r = 0

    if scan:
        for x1 in range(0, bins-3):
            for x2 in range(x1+1, bins-2):
                for x3 in range(x2+1, bins-1):
                    for x4 in range(x3+1, bins):
                        ratios = get_ratios(inputs, (x1, x2, x3, x4))
                        slope, intercept, r, p, std = linregress(ages, ratios)

                        if slope > 0 and r**2 >= best_p_r:
                            best_p_r = r**2
                            best_p_xs = x1, x2, x3, x4
                            print("p-%s-%s-%s-%s: %s" % (x1, x2, x3, x4, r**2))
                            correlation_plot(ratios, ages, "ratios-age-p-%s" % tag)

                        if slope < 0 and r**2 >= best_n_r:
                            best_n_r = r**2
                            best_n_xs = x1, x2, x3, x4
                            print("n-%s-%s-%s-%s: %s" % (x1, x2, x3, x4, r**2))
                            correlation_plot(ratios, ages, "ratios-age-n-%s" % tag)

    ratios1_training = get_ratios(inputs, best_p_xs)
    ratios2_training = get_ratios(inputs, best_n_xs)
    ratios_training = [[r1, r2] for r1, r2 in zip(ratios1_training, ratios2_training)]

    predictor = Lasso(alpha=0)
    predicted = cross_val_predict(predictor, ratios_training, ages, cv=5)

    predictor.fit(ratios_training, ages)
    print("-"*100)
    print(tag)
    print(predictor.score(ratios_training, ages))
    print("mse")
    print(mean_squared_error(ages, predicted))

    correlation_plot(predicted, ages, "ratios-predicted-actual-%s" % tag, line=True)

    predictor.best_p = best_p_xs
    predictor.best_n = best_n_xs

    return predictor


def get_ratios(inputs, xs):
    return [
        get_ratio(d, xs)
        for d in inputs
    ]


def get_ratio(data, xs):
    return np.sum(data[xs[0]:xs[1]]) / np.sum(data[xs[2]:xs[3]])


def correlation_plot(features, ages, filename, y_label='PCA-1', line=False):
    plt.figure()
    plt.scatter(ages, features)
    if line:
        plt.plot(np.linspace(15, 100), np.linspace(15, 100))
    plt.xlabel('Age')
    plt.ylabel(y_label)
    plt.xlim([15, 95])
    plt.title(r'$\mathrm{{{0} as function of age}}$'.format(y_label))
    plt.grid(True)
    plt.savefig("plots/{}.pdf".format(filename))


def print_progress(s):
    sys.stdout.write("\r%s" % s)
    sys.stdout.flush()
