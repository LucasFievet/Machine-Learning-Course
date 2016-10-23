import os
import numpy as np
import pandas as pd
import scipy.io

from .get_number_of_files import get_number_of_files
from .load_data_3d import load_sample_input, load_targets
from .settings import DATA_DIRECTORY

def create_deviation_set():
    mean = mean_brain()

    files = "train"
    DIR_OUT = os.path.join(
        DATA_DIRECTORY,
        "set_{}_deviation".format(files) 
        )
    if not os.path.exists(DIR_OUT):
        os.makedirs(DIR_OUT)
    nof = get_number_of_files(True) 
    for i in range(1,nof+1):
        data = load_sample_input(i,True)
        diff = brain_deviation(data, mean)
        FILE_OUT = os.path.join(
                DIR_OUT,
                "{}_{}.mat".format(files,i)
                )
        scipy.io.savemat(FILE_OUT, mdict={'out': diff}, oned_as='row')

    files = "test"
    DIR_OUT = os.path.join(
        DATA_DIRECTORY,
        "set_{}_deviation".format(files) 
        )
    if not os.path.exists(DIR_OUT):
        os.makedirs(DIR_OUT)
    nof = get_number_of_files(False) 
    for i in range(1,nof+1):
        data = load_sample_input(i,False)
        diff = brain_deviation(data, mean)
        FILE_OUT = os.path.join(
                DIR_OUT,
                "{}_{}.mat".format(files,i)
                )
        scipy.io.savemat(FILE_OUT, mdict={'out': diff}, oned_as='row')

def brain_deviation(data, mean):
    return data-mean

def mean_brain():
    targets = load_targets()
    targets = targets['Y'].tolist()
    nof = get_number_of_files(True)
    mean = np.zeros(load_sample_input(1,True).shape)
    for i in range(0,nof):
        data = load_sample_input(i+1,True) 
        mean += np.divide(data, targets.count(targets[i]))
    return np.divide(mean, len(set(targets)))



