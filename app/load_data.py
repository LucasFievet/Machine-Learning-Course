"""Loading training and testing data."""

import os

import nibabel as ni
import pandas as pd

from .settings import DATA_DIRECTORY

__author__ = "lfievet"
__copyright__ = "Copyright 2016, Project One"
__credits__ = ["lfievet"]
__license__ = "No License"
__version__ = "1.0"
__maintainer__ = "lfievet"
__email__ = "lfievet@ethz.ch"
__date__ = "12/10/2016"
__status__ = "Production"


def load_samples_inputs(training=True):
    inputs = []
    for i in range(1, 279):
        inputs.append(load_sample_input(i))

    return inputs


def load_sample_input(id=1, training=True):
    file_path = os.path.join(
        DATA_DIRECTORY,
        "set_train",
        "train_{}.nii".format(id)
    )
    return ni.load(file_path)


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
