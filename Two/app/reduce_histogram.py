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

    def __init__(self, bins, size):
        self.__bins = bins
        self.__size = size
        self.__test_set = load_samples_inputs(False)
        self.__train_set = load_samples_inputs(True)
        self.__box_positions()
        self.__len_train = len(self.__train_set)
        self.__len_test = len(self.__test_set)

        self.__train_path = os.path.join(CACHE_DIRECTORY, 'reduced_data', 'train')
        self.__test_path = os.path.join(CACHE_DIRECTORY, 'reduced_data', 'test')
        if not os.path.exists(self.__train_path):
            os.makedirs(self.__train_path)
            self.__compute_new('train')
        #if not os.path.exists(self.__test_path):
        #    os.makedirs(self.__test_path)
        #    self.__compute_new('test')

    def get_reduced_set(self, typ='train'):
        r = range(self.__len_train) if typ is 'train' else range(self.__len_test)
        return np.stack([self.get_reduced(i, typ) for i in r])

    def get_reduced(self, index, typ='train'):
        dir_path = self.__train_path if typ is 'train' else self.__test_path
        file_path = os.path.join(dir_path, '{}_{}.hdf'.format(typ, index))
        return np.array(pd.read_hdf(file_path, 'table'))

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
        for x , y, z in itertools.product(*self.__steps):
            out.append(self.__histogram(data.get_data()[x[0]:x[1], y[0]:y[1], z[0]:z[1], 0]))
        return out

    def __histogram(self, data):
        hist, edges = np.histogram(data, bins=self.__bins, range=[100, 1600], density=False)
        return hist

    def __box_positions(self):
        shape = np.shape(self.__train_set[0].get_data()[:, :, :, 0])
        s = self.__size
        self.__steps = list(map(lambda x: list(zip(range(0, x, s), range(s, x+s, s))), shape))
