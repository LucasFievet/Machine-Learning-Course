"""Description of this file."""

import matplotlib.pyplot as plt

import numpy as np
from .load_data import load_targets, load_samples_inputs

from .histogram_plot import histogram_plot


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

    # print(training_inputs[0].get_data().flatten().min())

    # volumes = [
    #     np.sum((i.get_data()[:, :, :, 0].flatten() > 120))
    #     for i in training_inputs
    # ]
    #
    # print(volumes)
    # volume_mean = np.mean(volumes)
    # volume_std = np.std(volumes)
    #
    # volumes -= volume_mean
    #
    # ages_filtered = []
    # volumes_filtered = []
    # for a, v in zip(ages, volumes):
    #     if abs(v) < 5*volume_std:
    #         ages_filtered.append(a)
    #         volumes_filtered.append(v)
    #
    # histogram_plot(volumes_filtered, "volumes")
    # correlation_plot(volumes_filtered, ages_filtered, "volume-age")

    half_x = len(training_inputs[0].get_data()[:, 0, 0, 0])/2
    half_y = len(training_inputs[0].get_data()[0, :, 0, 0])/2
    half_z = len(training_inputs[0].get_data()[0, 0, :, 0])/2

    ratios_x = [
        np.mean(i.get_data()[:half_x, :, :, 0]) / np.mean(i.get_data()[half_x:, :, :, 0])
        for i in training_inputs
    ]
    ratios_y = [
        np.mean(i.get_data()[:, :half_y, :, 0]) / np.mean(i.get_data()[:, half_y:, :, 0])
        for i in training_inputs
    ]
    ratios_z = [
        np.mean(i.get_data()[:, :, :half_z, 0]) / np.mean(i.get_data()[:, :, half_z:, 0])
        for i in training_inputs
    ]

    histogram_plot(ratios_x, "ratios-x")
    correlation_plot(ratios_x, ages, "ratios-x-age")

    histogram_plot(ratios_y, "ratios-y")
    correlation_plot(ratios_y, ages, "ratios-y-age")

    histogram_plot(ratios_z, "ratios-z")
    correlation_plot(ratios_z, ages, "ratios-z-age")


def correlation_plot(features, ages, filename, y_label='PCA-1'):
    plt.figure()
    plt.scatter(ages, features)
    plt.xlabel('Age')
    plt.ylabel(y_label)
    plt.xlim([15, 95])
    # plt.ylim([0, 1])
    plt.title(r'$\mathrm{{{0} as function of age}}$'.format(y_label))
    plt.grid(True)
    plt.savefig("plots/{}.pdf".format(filename))
