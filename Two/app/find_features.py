"""Description of this file."""

import os
import warnings
import itertools

import scipy.io

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf
from mpl_toolkits.mplot3d import Axes3D

from .load_data import load_targets
from .settings import CACHE_DIRECTORY, PLOT_DIRECTORY
from .reduce_histogram import ReduceHistogram

warnings.filterwarnings("ignore", category=DeprecationWarning)

class FindFeatures():
    " TODO doc "
    def __init__(self, bins=10, size=20, thresh=250):
        data = ReduceHistogram(bins, size)
        self.__data = np.transpose(data.get_reduced_set('train'), (2, 1, 0))
        self.__train = data.get_reduced_set('train')
        self.__test = data.get_reduced_set('test')
        self.__targets = load_targets()['Y'].tolist()
        self.__evaluated = np.array([])
        self.__evaluate_features()
        self.__thresh = thresh
        self.__locations = self.__exctract_significant()

    def get_significant(self, typ='train'):
        """ doc
        """
        return np.array(self.__exctract_values(typ))

    def __exctract_values(self, typ='train'):
        data = self.__train if typ is 'train' else self.__test
        loc = self.__locations
        range_d = range(np.shape(data)[0])
        range_l = range(np.shape(loc)[0])
        return [[data[d, loc[l, 1], loc[l, 0]] for l in range_l] for d in range_d]


    def __exctract_significant(self):
        evaluated = self.__evaluated
        shape = evaluated[:, :, 0].shape
        mean_dif = np.fabs(evaluated[:, :, 0]-evaluated[:, :, 2])
        var_avg = np.sqrt(np.divide((evaluated[:, :, 1]+evaluated[:, :, 3]), 2))
        iteration = [range(shape[0]), range(shape[1])]
        locations = [c for c in itertools.product(*iteration)
                     if mean_dif[c[0], c[1]] > self.__thresh
                     if mean_dif[c[0], c[1]] > var_avg[c[0], c[1]]]
        print('Number of significant areas found:', np.shape(locations)[0])
        return np.array(locations)

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
            plt.figure(num=i+1)
            plt.errorbar(x_range, np.fabs(data[0, :]-data[2, :]),
                         np.sqrt(np.fabs(data[1, :]-data[3, :])),
                         linestyle='None', marker='*', markersize=2.0,
                         linewidth=0.7, capsize=1)
        path = os.path.join(PLOT_DIRECTORY, 'plot_mean_var.pdf')
        pdf = backend_pdf.PdfPages(path)
        for fig in range(1, plt.figure().number):
            pdf.savefig(fig)
        plt.close('all')
        pdf.close()

    def plot_mean(self):
        """TODO: Docstring for plot_mean.
        """
        length = len(self.__evaluated[:, 0, 0])
        for i in range(length):
            data = self.__evaluated[i, :, :].transpose()
            x_range = range(len(data[0, :]))
            plt.figure(num=i+1)
            plt.plot(x_range, np.fabs(data[0, :]-data[2, :]))

        path = os.path.join(PLOT_DIRECTORY, "plot_mean.pdf")
        pdf = backend_pdf.PdfPages(path)
        for fig in range(1, plt.figure().number):
            pdf.savefig(fig)
        plt.close('all')
        pdf.close()

    def plot_var(self):
        """TODO: Docstring for plot_var.
        """
        length = len(self.__evaluated[:, 0, 0])
        for i in range(length):
            data = self.__evaluated[i, :, :].transpose()
            x_range = range(len(data[0, :]))
            plt.figure(num=i+1)
            plt.scatter(x_range, np.sqrt(np.fabs(data[1, :])),
                        s=3, c='r', marker='*', edgecolors='none')
            plt.scatter(x_range, np.sqrt(np.fabs(data[3, :])),
                        s=3, c='b', marker='^', edgecolors='none')

        path = os.path.join(PLOT_DIRECTORY, "plot_var.pdf")
        pdf = backend_pdf.PdfPages(path)
        for fig in range(1, plt.figure().number):
            pdf.savefig(fig)
        plt.close('all')
        pdf.close()

    def plot_significant(self):
        """ plot_significant
        """
        locations = self.__locations
        mean_dif = np.fabs(self.__evaluated[:, :, 0]-self.__evaluated[:, :, 2])
        z_range = range(np.shape(locations)[0])
        z_val = np.array([mean_dif[locations[i, 0], locations[i, 1]] for i in z_range])

        fig = plt.figure()
        axi = fig.add_subplot(111, projection='3d')
        axi.scatter(locations[:, 0], locations[:, 1], z_val)
        axi.set_xlabel('bins')
        axi.set_ylabel('box')
        axi.set_zlabel('mean difference')
        plt.show()
        plt.close(fig)


    def test(self):
        """ TEST
        """
        self.plot_significant()
