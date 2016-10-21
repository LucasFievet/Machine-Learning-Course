import os
import scipy.io

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

