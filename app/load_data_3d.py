"""Loading training and testing data."""

import os

import nibabel as ni
import pandas as pd

from .settings import DATA_DIRECTORY

def load_samples_inputs(training=True):
    DIR = os.path.join(
        DATA_DIRECTORY,
        "set_train"
        )
    files = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

    inputs = []
    for i in range(1, files+1):
        inputs.append(load_sample_input(i))

    return inputs


def load_sample_input(id=1, training=True):
    file_path = os.path.join(
        DATA_DIRECTORY,
        "set_train",
        "train_{}.nii".format(id)
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
