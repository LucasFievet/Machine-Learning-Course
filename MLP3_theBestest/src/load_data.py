"""Loading training and testing data."""

import os

import nibabel as ni
import pandas as pd

from .settings import DATA_DIRECTORY

def load_samples_inputs(training=True):
    inputs = []
    for i in range(1, 279 if training else 139):
        inputs.append(load_sample_input(i, training))
    return inputs

def load_sample_input(index=1, training=True):
    tag = "train" if training else "test"
    file_path = os.path.join(DATA_DIRECTORY, "set_{}".format(tag),
                             "{}_{}.nii".format(tag, index))
    return ni.load(file_path)

def load_targets():
    targets_path = os.path.join(DATA_DIRECTORY, "targets.csv")
    return pd.read_csv(targets_path, header=None, names=["G", "A", "H"])
