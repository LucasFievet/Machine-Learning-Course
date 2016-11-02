import os

import pandas as pd
import numpy as np


CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DIR = os.path.join(CURRENT_DIRECTORY,"iterate")
files = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
for i in range(0,files):
    path = os.path.join(DIR,"se_{}.hdf".format(i))
    if i == 0:
        data = pd.read_hdf(path, "table").values
    else:
        data = np.concatenate((data,pd.read_hdf(path, "table").values), axis=0)

d = data.tolist()
d.sort(key=lambda x: x[0])
for i in range(0,10):
    print(d[i])
