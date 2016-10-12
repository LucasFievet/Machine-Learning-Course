"""Loading training and testing data."""

import os

import nibabel as ni

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


def load_data(training=True):
    file_path = os.path.join(DATA_DIRECTORY, "set_train", "train_1.nii")
    data = ni.load(file_path)

    # print(data)
    return data
