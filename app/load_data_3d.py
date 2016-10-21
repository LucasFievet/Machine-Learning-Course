"""Loading training and testing data."""

import os
import scipy.io

import nibabel as ni
import pandas as pd

from .settings import DATA_DIRECTORY
from .mean_brain import create_deviation_set 
from .get_number_of_files import get_number_of_files

def load_deviations(training=True):
    if training == True:
        files = "train"
    else:
        files = "test"
    DIR = os.path.join(
        DATA_DIRECTORY,
        "set_{}_deviation".format(files),
        )
    if not os.path.exists(DIR):
        create_deviation_set(training)
    data = []
    for i in range(1,get_number_of_files(training)+1):
        data.append(load_deviation(i,training))
    return data

def load_deviation(ID,training=True):
    if training == True:
        files = "train"
    else:
        files = "test"
    FILE = os.path.join(
        DATA_DIRECTORY,
        "set_{}_deviation".format(files),
        "{}_{}.mat".format(files,ID)
        )
    return scipy.io.loadmat(FILE)

def load_samples_inputs(training=True):
    nof = get_number_of_files(training) 
    inputs = []
    for i in range(1, nof+1):
        inputs.append(load_sample_input(i, training))
    return inputs

def load_sample_input(id=1, training=True):
    if training == True:
        folder = "set_train"
        files = "train"
    else:
        folder = "set_test"
        files = "test"

    file_path = os.path.join(
        DATA_DIRECTORY,
        folder,
        "{}_{}.nii".format(files,id)
    )
    return ni.load(file_path).get_data()[:,:,:,0]


def load_targets(training=True):
    targets_path = os.path.join(
        DATA_DIRECTORY,
        "targets.csv"
    )
    return pd.read_csv(
        targets_path,
        header=None,
        names=["Y"]
    )
