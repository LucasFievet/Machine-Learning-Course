"""Description of this file."""

import os
import warnings
import itertools

import numpy as np
import pandas as pd

from .load_data import load_samples_inputs
from .settings import CACHE_DIRECTORY

warnings.filterwarnings("ignore", category=DeprecationWarning)

class ReduceHistogram:

    """Docstring for reduce_histogram. """

    def __init__(self, bin_size, box_size):
        self.__size = box_size
        self.__test_set = load_samples_inputs(False)
        self.__train_set = load_samples_inputs(True)
        self.__len_train = len(self.__train_set)
        self.__len_test = len(self.__test_set)
        self.__max = self.__find_max()
        self.__bins = self.__compute_bins(bin_size)
        self.__box_positions()
        self.__train_path = os.path.join(CACHE_DIRECTORY, 'reduced_data',
                                         'train_{}_{}'.format(bin_size, box_size))
        self.__test_path = os.path.join(CACHE_DIRECTORY, 'reduced_data',
                                        'test_{}_{}'.format(bin_size, box_size))

    def get_reduced_set(self, typ='train'):
        """ doc
        """
        self.__check_exists(typ)
        rge = range(self.__len_train) if typ is 'train' else range(self.__len_test)
        return np.stack([self.__get_reduced(i, typ) for i in rge])

    def get_reduced(self, index, typ='train'):
        """ doc
        """
        self.__check_exists(typ)
        return self.__get_reduced(index, typ)

    def __get_reduced(self, index, typ='train'):
        dir_path = self.__train_path if typ is 'train' else self.__test_path
        file_path = os.path.join(dir_path, '{}_{}.hdf'.format(typ, index))
        return np.array(pd.read_hdf(file_path, 'table'))

    def __check_exists(self, typ='train'):
        if typ == 'train':
            if not os.path.exists(self.__train_path):
                os.makedirs(self.__train_path)
                self.__compute_new('train')
        else:
            if not os.path.exists(self.__test_path):
                os.makedirs(self.__test_path)
                self.__compute_new('test')

    def __compute_new(self, typ='train'):
        data = self.__train_set if typ is 'train' else self.__test_set
        for index, d in enumerate(data):
            reduced = self.__reduce(d)
            self.__save_reduced(reduced, index, typ)

    def __save_reduced(self, data, index, typ):
        dir_path = self.__train_path if typ is 'train' else self.__test_path
        file_path = os.path.join(dir_path, '{}_{}.hdf'.format(typ, index))
        pd.DataFrame(data).to_hdf(file_path, 'table')

    def __reduce(self, data):
        out = []
        for x, y, z in itertools.product(*self.__steps):
            out.append(self.__histogram(data.get_data()[x[0]:x[1], y[0]:y[1], z[0]:z[1], 0]))
        return out

    def __histogram(self, data):
        hist, edges = np.histogram(data, bins=self.__bins, range=[100, self.__max], density=False)
        return hist

    def __box_positions(self):
        shape = np.shape(self.__train_set[0].get_data()[:, :, :, 0])
        s = self.__size
        self.__steps = list(map(lambda x: list(zip(range(0, x, s), range(s, x+s, s))), shape))

    def __compute_bins(self, bin_size):
        bins = self.__max/bin_size
        bins = int(round(bins))
        print('number of bins:', bins)
        return bins

    def __find_max(self):
        """
        file_path = os.path.join(CACHE_DIRECTORY, 'max_val.hdf')
        if os.path.exists(file_path):
            return pd.read_hdf(file_path, 'table')[0]
        else:
            max_train = [np.amax(i.get_data()[:, :, :, 0]) for i in self.__train_set]
            max_test = [np.amax(i.get_data()[:, :, :, 0]) for i in self.__test_set]
            max_train = np.amax(max_train)
            max_test = np.amax(max_test)
            maxima = np.amax([max_train, max_test])
            pd.DataFrame([maxima]).to_hdf(file_path, 'table')
            print('Maxmal Value of:', maxima)
            return maxima
        """
        return 4418

    def test(self):
        print(self.__find_max())
