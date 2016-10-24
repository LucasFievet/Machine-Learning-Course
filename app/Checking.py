"""For checking various modules."""

import os

import nibabel as ni
import pandas as pd
import numpy as np

from .settings import DATA_DIRECTORY
from .load_data import load_sample_input


def checkdata():
    sample1 = 20
    data1 = load_sample_input(sample1)
    print data1.shape
    print type(data1)
    return
