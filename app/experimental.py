"""Description of this file."""

import os
import sys

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from .load_data import load_targets, load_samples_inputs
from .predictor_cluster import histogram_plot

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, LinearRegression
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

    # inputs_20_25, young_ages = filter_samples(training_inputs, ages, 25, 60)
    # brain_variance(inputs_20_25)
    #
    # return

    young_inputs, young_ages = filter_samples(training_inputs, ages, 0, 60)
    old_inputs, old_ages = filter_samples(training_inputs, ages, 60, 100)

    predictor_all = cross_val_predict_data2(training_inputs, ages, "all")
    predictor_young = cross_val_predict_data2(young_inputs, young_ages, "young")
    predictor_old = cross_val_predict_data2(old_inputs, old_ages, "old")
    # print(np.var(young_ages))

    # return

    test_inputs = load_samples_inputs(False)
    test_data = pre_process(test_inputs)
    test_data = [binarize(i, 7) for i in test_data]

    # ratios1_test = get_ratios(test_data, predictor_all.best_p)
    # ratios2_test = get_ratios(test_data, predictor_all.best_n)
    # ratios_test = [[r1, r2] for r1, r2 in zip(ratios1_test, ratios2_test)]

    test_predicted = predictor_all.predict(test_data)

    for idx, p in enumerate(test_predicted):
        data_idx = test_data[idx]
        if p < 50:
            refined_prediction = predictor_young.predict(data_idx)
            test_predicted[idx] = refined_prediction
            print("{}: {}".format(p, refined_prediction))
        if p >= 60:
            refined_prediction = predictor_old.predict(data_idx)
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
        if age_min < age <= age_max:
            filtered_inputs.append(input_data)
            filtered_ages.append(age)

    return filtered_inputs, filtered_ages


def cross_val_predict_data(inputs, ages, tag="all"):
    bins = 30
    scan = True

    if tag == "all":
        scan = False
        best_p_xs = 9, 15, 18, 19
        best_n_xs = 12, 13, 15, 24
    elif tag == "young":
        scan = True
        best_p_xs = 0, 5, 24, 29
        best_n_xs = 12, 14, 16, 27
    elif tag == "old":
        scan = False
        best_p_xs = 0, 7, 23, 29
        best_n_xs = 0, 4, 13, 14
    else:
        best_p_xs = 0, 0, 0, 0
        best_n_xs = 0, 0, 0, 0

    inputs = pre_process(inputs)
    inputs = [binarize(i, bins) for i in inputs]
    best_mse = 100
    best_p_r = 0
    best_n_r = 0

    predictor = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression(fit_intercept=False))
    ])

    if scan:
        for x1 in range(0, bins-3):
            for x2 in range(x1+1, bins-2):
                for x3 in range(x2+1, bins-1):
                    for x4 in range(x3+1, bins):
                        ratios = get_ratios(inputs, (x1, x2, x3, x4))
                        ratios = [[r] for r in ratios]
                        predicted = cross_val_predict(predictor, ratios, ages, cv=10, n_jobs=4)
                        mse = mean_squared_error(ages, predicted)
                        print_progress("%s-%s-%s-%s: %s" % (x1, x2, x3, x4, mse))

                        if mse < best_mse:
                            best_mse = mse
                            print("\n%s-%s-%s-%s: %s" % (x1, x2, x3, x4, mse))
                            best_p_xs = x1, x2, x3, x4

                        # slope, intercept, r, p, std = linregress(ages, ratios)
                        # coefs, r, rank, singular_values, rcond = np.polyfit(ratios, ages, 2, full=True)
                        # slope = coefs[1]
                        # r = 1 - (r[0]/len(ages))/np.var(ages)
                        #
                        # if slope > 0 and r**2 > best_p_r:
                        #     best_p_r = r**2
                        #     best_p_xs = x1, x2, x3, x4
                        #     print_progress("p-%s-%s-%s-%s: %s" % (x1, x2, x3, x4, r**2))
                        #     correlation_plot(ratios, ages, "ratios-age-p-%s" % tag)
                        #
                        # if slope < 0 and r**2 > best_n_r:
                        #     best_n_r = r**2
                        #     best_n_xs = x1, x2, x3, x4
                        #     print_progress("n-%s-%s-%s-%s: %s" % (x1, x2, x3, x4, r**2))
                        #     correlation_plot(ratios, ages, "ratios-age-n-%s" % tag)

    ratios1_training = get_ratios(inputs, best_p_xs)
    ratios2_training = get_ratios(inputs, best_n_xs)
    ratios_training = [[r1, r2] for r1, r2 in zip(ratios1_training, ratios2_training)]
    # ratios = get_ratios(inputs, best_p_xs)
    # ratios_training = [[r] for r in ratios]

    # predictor = Lasso(alpha=0)
    predicted = cross_val_predict(predictor, ratios_training, ages, cv=10)

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


def cross_val_predict_data2(inputs, ages, tag="all"):
    bins = 7
    inputs = pre_process(inputs)
    inputs = [binarize(i, bins) for i in inputs]

    predictor = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression(fit_intercept=False))
    ])

    predicted = cross_val_predict(predictor, inputs, ages, cv=10, n_jobs=6)

    predictor.fit(inputs, ages)
    print("-"*100)
    print(tag)
    print(predictor.score(inputs, ages))
    print("mse")
    print(mean_squared_error(ages, predicted))

    correlation_plot(predicted, ages, "ratios-predicted-actual-%s" % tag, line=True)

    return predictor


def binarize(i, bins):
    hist, edges = np.histogram(i, bins=bins, range=[100, 1600], normed=True)
    edges = (edges[:-1] + edges[1:])/2
    hist *= edges
    return hist


def pre_process(data):
    inputs = np.array([i.get_data()[:, :, :, 0].flatten() for i in data])
    return inputs
    stds = np.std(inputs, axis=0)
    # histogram_plot(stds[stds > 0], "stds-hist", bins=100)

    indices = np.where(stds < 75)
    return [i[indices] for i in inputs]
    # histogram_plot(inputs, "stds-max-hist", bins=10)
    # print(np.where(stds > 300))


    # inputs = [np.histogram(i, bins=bins, range=[100, 1600])[0] for i in inputs]


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
