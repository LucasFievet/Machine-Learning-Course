"""Description of this file."""

import os

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
    l = -1
    training_inputs = load_samples_inputs()[:l]
    data = load_targets()
    ages = data["Y"].tolist()[:l]

    data = [i.get_data()[:, :, :, 0].flatten() for i in training_inputs]
    data = [np.histogram(i, bins=30, range=[100, 1600])[0] for i in data]
    best_p_r = 0
    best_p_x1, best_p_x2, best_p_x3, best_p_x4 = 8, 9, 10, 11
    best_n_r = 0
    best_n_x1, best_n_x2, best_n_x3, best_n_x4 = 12, 14, 15, 23

    for x1 in range(0, 28):
        for x2 in range(x1+1, 28):
            for x3 in range(x2+1, 29):
                for x4 in range(x3+1, 30):
                    ratios = [
                        np.sum(d[x1:x2]) / np.sum(d[x3:x4])
                        for d in data
                    ]

                    slope, intercept, r, p, std = linregress(ages, ratios)

                    if slope > 0 and r**2 >= best_p_r:
                        best_p_r = r**2
                        best_p_x1, best_p_x2, best_p_x3, best_p_x4 = x1, x2, x3, x4
                        print("p-%s-%s-%s-%s: %s" % (x1, x2, x3, x4, r**2))
                        correlation_plot(ratios, ages, "ratios-age-p-%s" % best_p_r)

                    if slope < 0 and r**2 >= best_n_r:
                        best_n_r = r**2
                        best_n_x1, best_n_x2, best_n_x3, best_n_x4 = x1, x2, x3, x4
                        print("n-%s-%s-%s-%s: %s" % (x1, x2, x3, x4, r**2))
                        correlation_plot(ratios, ages, "ratios-age-n-%s" % best_n_r)

    ratios_training = [
        [
            np.sum(d[best_p_x1:best_p_x2]) / np.sum(d[best_p_x3:best_p_x4]),
            np.sum(d[best_n_x1:best_n_x2]) / np.sum(d[best_n_x3:best_n_x4])
        ]
        for d in data
    ]

    predictor = Lasso(alpha=0)
    predicted = cross_val_predict(predictor, ratios_training, ages, cv=5)

    predictor.fit(ratios_training, ages)
    print()
    print(predictor.score(ratios_training, ages))
    print("mse")
    print(mean_squared_error(ages, predicted))

    correlation_plot(predicted, ages, "ratios-predicted-actual")

    test_inputs = load_samples_inputs(False)
    test_data = [i.get_data()[:, :, :, 0].flatten() for i in test_inputs]
    test_data = [np.histogram(i, bins=30, range=[100, 1600])[0] for i in test_data]

    ratios_test = [
        [
            np.sum(d[best_p_x1:best_p_x2]) / np.sum(d[best_p_x3:best_p_x4]),
            np.sum(d[best_n_x1:best_n_x2]) / np.sum(d[best_n_x3:best_n_x4])
        ]
        for d in test_data
    ]

    test_predicted = predictor.predict(ratios_test)
    test_predicted = [20 if p < 18 else p for p in test_predicted]
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


def correlation_plot(features, ages, filename, y_label='PCA-1'):
    plt.figure()
    plt.scatter(ages, features)
    plt.plot(np.linspace(15, 100), np.linspace(15, 100))
    plt.xlabel('Age')
    plt.ylabel(y_label)
    plt.xlim([15, 95])
    plt.title(r'$\mathrm{{{0} as function of age}}$'.format(y_label))
    plt.grid(True)
    plt.savefig("plots/{}.pdf".format(filename))
