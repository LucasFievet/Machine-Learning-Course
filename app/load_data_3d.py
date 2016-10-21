"""Loading training and testing data."""
import os
import scipy.io

import nibabel as ni
import pandas as pd

from .settings import DATA_DIRECTORY
from .get_number_of_files import get_number_of_files

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
