import os
import numpy as np
import pandas as pd
import scipy.io

from .get_number_of_files import get_number_of_files
from .load_data_3d import load_sample_input
from .settings import DATA_DIRECTORY

def create_deviation_set(training=True):
    if training == True:
        folder = "set_train"
        files = "train"
    else:
        folder = "set_test"
        files = "test"
    DIR_IN = os.path.join(
        DATA_DIRECTORY,
        folder
        )
    DIR_OUT = os.path.join(
        DATA_DIRECTORY,
        "{}_deviation".format(folder) 
        )
    if not os.path.exists(DIR_OUT):
        os.makedirs(DIR_OUT)

    mean = mean_brain(training)
    nof = get_number_of_files(training) 
    for i in range(1,nof+1):
        data = load_sample_input(i,training)
        diff = brain_deviation(data, mean, training)
        FILE_OUT = os.path.join(
                DIR_OUT,
                "{}_{}.mat".format(files,i)
                )
        scipy.io.savemat(FILE_OUT, mdict={'out': diff}, oned_as='row')


def brain_deviation(data, mean=None, training=True):
    if mean == None:
        mean = mean_brain(training)
        return (data-mean), mean
    else:
        return data-mean

def mean_brain(training=True):
    nof = get_number_of_files(training)
    mean = np.zeros(load_sample_input(1,training).shape)
    for i in range(1,nof+1):
        data = load_sample_input(i,training) 
        mean += np.divide(i,nof)
    return mean



