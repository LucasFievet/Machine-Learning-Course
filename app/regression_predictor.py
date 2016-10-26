"""Description of this file."""

import numpy as np
from .load_data import load_targets, load_samples_inputs

__author__ = "lfievet"
__copyright__ = "Copyright 2016, Project One"
__credits__ = ["lfievet"]
__license__ = "No License"
__version__ = "1.0"
__maintainer__ = "lfievet"
__email__ = "lfievet@ethz.ch"
__date__ = "26/10/2016"
__status__ = "Production"


def regression_predictor():
    training_inputs = load_samples_inputs()
    data = load_targets()
    ages = data["Y"].tolist()