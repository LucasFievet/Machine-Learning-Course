import numpy as np
import matplotlib.pyplot as plt

from .load_data_3d import load_targets
from .settings import DATA_DIRECTORY

def age_hist():
    data = load_targets()
    data = data['Y'].tolist()
    plt.figure()
    plt.hist(data, bins=100)
    plt.savefig("plots/age_hist.pdf")
    plt.close()
