"""Description of this file."""

import os
import warnings

import itertools

import numpy as np
import pandas as pd
import scipy.io

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

    def __init__(self, bins=10, size=20):
        data = ReduceHistogram(bins, size).get_reduced_set('train')
        self.__data = np.transpose(data, (2,1,0))
        self.__targets = load_targets()['Y'].tolist()
    
    def __evaluate_features(self):
        FILE = os.path.join(CACHE_DIRECTORY, 'evaluate_features.mat')
        if os.path.exists(FILE):
            self.__evaluated = scipy.io.loadmat(FILE)['data']
        else:
            shape = self.__data[:,:,0].shape
            intervals = (range(shape[0]), range(shape[1]))
            evaluated = [self.__eval_fun(self.__data[x,y,:]) 
                    for x, y in itertools.product(*intervals)]
            l = len(self.__eval_fun(self.__data[0,0,:]))
            self.__evaluated = np.array(evaluated).reshape(shape[0], shape[1], l)
            scipy.io.savemat(FILE, mdict={'data': self.__evaluated}, oned_as='row')
        print('shape of evaluated data:', np.shape(self.__evaluated))

    def __eval_fun(self, data):
        tmp = list(zip(self.__targets, data))
        sick = [x[1] for x in tmp if x[0]==1]
        healthy = [x[1] for x in tmp if x[0]==0]
        sick_mean = np.mean(sick)
        sick_var = np.var(sick)
        healthy_mean = np.mean(healthy)
        healthy_var = np.var(healthy)
        return np.array([sick_mean, sick_var, healthy_mean, healthy_var])
        
    def test(self):
        self.__evaluate_features()

        
