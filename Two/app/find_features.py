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

from .load_data import load_samples_inputs, load_targets

from .settings import CURRENT_DIRECTORY, CACHE_DIRECTORY
from .reduce_histogram import ReduceHistogram

warnings.filterwarnings("ignore", category=DeprecationWarning)

class FindFeatures():

    def __init__(self, bins=10, size=50):
        data = ReduceHistogram(bins, size).get_reduced_set('train')
        self.__data = np.transpose(data, (2,1,0))
        self.__targets = load_targets()['Y'].tolist()
    
    def __evaluate_features(self):
        shape = self.__data[:,:,0].shape
        self.__evaluated = np.zeros(shape)
        intervals = (range(shape[0]), range(shape[1]))
        for x, y in itertools.product(*intervals):
            self.__evaluated[x,y] = self.__eval_fun(self.__data[x,y,:])  
        print(self.__evaluated)

    def __eval_fun(self, data):
        return 0
        
    def test(self):
        self.__evaluate_features()

        
