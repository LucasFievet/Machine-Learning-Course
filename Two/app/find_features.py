"""Description of this file."""

import os
import warnings
import itertools

import scipy.io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends import backend_pdf

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.metrics import log_loss, make_scorer

from .load_data import load_targets

from .settings import CACHE_DIRECTORY
from .reduce_histogram import ReduceHistogram

warnings.filterwarnings("ignore", category=DeprecationWarning)

class FindFeatures():
    " TODO doc "
    def __init__(self, bins=10, size=20):
        data = ReduceHistogram(bins, size).get_reduced_set('train')
        self.__data = np.transpose(data, (2, 1, 0))
        self.__targets = load_targets()['Y'].tolist()
        self.__evaluated = np.array([])

    def __evaluate_features(self):
        file_path = os.path.join(CACHE_DIRECTORY, 'evaluate_features.mat')
        if os.path.exists(file_path):
            self.__evaluated = scipy.io.loadmat(file_path)['data']
        else:
            shape = self.__data[:, :, 0].shape
            intervals = (range(shape[0]), range(shape[1]))
            evaluated = [self.__eval_fun(self.__data[x, y, :])
                         for x, y in itertools.product(*intervals)]
            length = len(self.__eval_fun(self.__data[0, 0, :]))
            self.__evaluated = np.array(evaluated).reshape(shape[0], shape[1], length)
            scipy.io.savemat(file_path, mdict={'data': self.__evaluated}, oned_as='row')
        print('shape of evaluated data:', np.shape(self.__evaluated))

    def __eval_fun(self, data):
        tmp = list(zip(self.__targets, data))
        sick = [x[1] for x in tmp if x[0] == 1]
        healthy = [x[1] for x in tmp if x[0] == 0]
        sick_mean = np.mean(sick)
        sick_var = np.var(sick)
        healthy_mean = np.mean(healthy)
        healthy_var = np.var(healthy)
        return np.array([sick_mean, sick_var, healthy_mean, healthy_var])

    def plot_mean_var(self):
        """TODO: Docstring for plot_mean_var.
        """
        length = len(self.__evaluated[:, 0, 0])
        for i in range(length):
            data = self.__evaluated[i, :, :].transpose()
            x_range = range(len(data[0, :]))
            plt.figure() 
            plt.errorbar(x_range, np.fabs(data[0, :]-data[2, :]), np.sqrt(np.fabs(data[1, :]-data[3, :])),
                         linestyle='None', marker='*', markersize=2.0, linewidth=0.7, capsize=1)

        pdf = backend_pdf.PdfPages("plot_mean_var.pdf")
        for fig in range(1, plt.figure().number): ## will open an empty extra figure :(
            pdf.savefig( fig )
        pdf.close()

    def test(self):
        ' DOC '
        self.__evaluate_features()
