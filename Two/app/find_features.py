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
    def __init__(self, bin_size=200, box_size=20, thresh=20, fac=2):
        data = ReduceHistogram(bin_size, box_size)
        self.__vars = {'fac': fac, 'thresh': thresh}
        self.__vars['data'] = np.transpose(data.get_reduced_set('train'), (2, 1, 0))
        self.__vars['train'] = data.get_reduced_set('train')
        self.__vars['test'] = data.get_reduced_set('test')
        self.__vars['targets'] = load_targets()['Y'].tolist()
        self.__vars['evaluated'] = self.__evaluate_features()
        self.__vars['locations'] = self.__exctract_significant()

    def get_significant(self, typ='train'):
        """ doc
        """
        return np.array(self.__exctract_values(typ))

    def get_targets(self):
        """ doc
        """
        return self.__vars['targets']

    def __exctract_values(self, typ='train'):
        data = self.__vars[typ]
        loc = self.__vars['locations']
        range_d = range(np.shape(data)[0])
        range_l = range(np.shape(loc)[0])
        return [[data[d, loc[l, 1], loc[l, 0]] for l in range_l] for d in range_d]


    def __exctract_significant(self):
        evaluated = self.__vars['evaluated']
        shape = evaluated[:, :, 0].shape
        mean_dif = np.fabs(evaluated[:, :, 0]-evaluated[:, :, 2])
        var_avg = np.sqrt(np.divide((evaluated[:, :, 1]+evaluated[:, :, 3]), self.__vars['fac']))
        iteration = [range(shape[0]), range(shape[1])]
        locations = [c for c in itertools.product(*iteration)
                     if mean_dif[c] > self.__vars['thresh']
                     if mean_dif[c] > var_avg[c]]
        print('Number of significant areas found:', np.shape(locations)[0])
        return np.array(locations)

    def __evaluate_features(self):
        file_path = os.path.join(CACHE_DIRECTORY, 'evaluate_features.mat')
        if os.path.exists(file_path):
            out = scipy.io.loadmat(file_path)['data']
        else:
            shape = self.__vars['data'][:, :, 0].shape
            intervals = (range(shape[0]), range(shape[1]))
            evaluated = [self.__eval_fun(self.__vars['data'][x, y, :])
                         for x, y in itertools.product(*intervals)]
            length = len(self.__eval_fun(self.__vars['data'][0, 0, :]))
            out = np.array(evaluated).reshape(shape[0], shape[1], length)
            scipy.io.savemat(file_path, mdict={'data': out}, oned_as='row')
        print('shape of evaluated data:', np.shape(out))
        return out

    def __eval_fun(self, data):
        tmp = list(zip(self.__vars['targets'], data))
        sick = [x[1] for x in tmp if x[0] == 1]
        healthy = [x[1] for x in tmp if x[0] == 0]
        sick_mean = np.mean(sick)
        sick_var = np.var(sick)
        healthy_mean = np.mean(healthy)
        healthy_var = np.var(healthy)
        return np.array([sick_mean, sick_var, healthy_mean, healthy_var])

    def plot_mean_var_diff(self):
        """TODO: Docstring for plot_mean_var.
        """
        width = 0.3
        length = len(self.__vars['evaluated'][:, 0, 0])
        path = os.path.join(PLOT_DIRECTORY, 'plot_mean_var_diff.pdf')
        pdf = backend_pdf.PdfPages(path)
        for i in range(length):
            data = self.__vars['evaluated'][i, :, :].transpose()
            x_range = range(len(data[0, :]))
            fig = plt.figure()
            fig.suptitle('bin {}'.format(i), fontsize=12)
            y_range = (np.fabs(data[0, :]-data[2, :])-np.sqrt(
                np.divide(data[1, :]+data[3, :], self.__vars['fac'])))
            plt.bar(x_range, y_range, width, color='black', linewidth=0)
            pdf.savefig(fig)
            plt.close(fig)
        pdf.close()

    def plot_mean_var(self):
        """TODO: Docstring for plot_mean_var.
        """
        width = 0.3
        length = len(self.__vars['evaluated'][:, 0, 0])
        path = os.path.join(PLOT_DIRECTORY, 'plot_mean_var.pdf')
        pdf = backend_pdf.PdfPages(path)
        for i in range(length):
            data = self.__vars['evaluated'][i, :, :].transpose()
            x_range = range(len(data[0, :]))
            fig = plt.figure()
            fig.suptitle('bin {}'.format(i), fontsize=12)
            plt.bar(x_range, np.fabs(data[0, :]-data[2, :]), width, color='g', linewidth=0)
            plt.bar(x_range, np.sqrt(np.divide(data[1, :]+data[3, :], self.__vars['fac'])),
                    width, color='r', linewidth=0)
            pdf.savefig(fig)
            plt.close(fig)
        pdf.close()

    def plot_mean(self):
        """TODO: Docstring for plot_mean.
        """
        length = len(self.__vars['evaluated'][:, 0, 0])
        path = os.path.join(PLOT_DIRECTORY, "plot_mean.pdf")
        pdf = backend_pdf.PdfPages(path)
        for i in range(length):
            data = self.__vars['evaluated'][i, :, :].transpose()
            x_range = range(len(data[0, :]))
            fig = plt.figure()
            fig.suptitle('bin {}'.format(i), fontsize=12)
            plt.bar(x_range, np.fabs(data[0, :]-data[2, :]), 0.3, color='black', linewidth=0)
            pdf.savefig(fig)
            plt.close(fig)
        pdf.close()

    def plot_var(self):
        """TODO: Docstring for plot_var.
        """
        length = len(self.__vars['evaluated'][:, 0, 0])
        path = os.path.join(PLOT_DIRECTORY, "plot_var.pdf")
        pdf = backend_pdf.PdfPages(path)
        for i in range(length):
            data = self.__vars['evaluated'][i, :, :].transpose()
            x_range = range(len(data[0, :]))
            fig = plt.figure()
            plt.scatter(x_range, np.sqrt(np.fabs(data[1, :])),
                        s=3, c='r', marker='*', edgecolors='none')
            plt.scatter(x_range, np.sqrt(np.fabs(data[3, :])),
                        s=3, c='b', marker='^', edgecolors='none')
            pdf.savefig(fig)
            plt.close(fig)
        pdf.close()

    def plot_significant(self):
        """ plot_significant
        """
        locations = self.__vars['locations']
        mean_dif = np.fabs(self.__vars['evaluated'][:, :, 0]-self.__vars['evaluated'][:, :, 2])
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
