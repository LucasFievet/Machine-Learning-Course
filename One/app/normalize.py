import numpy as np

def normalize(inputs):
    data = np.array(inputs)
    maximum = data.max()
    minimum = data.min()

    out = (data-minimum)/(maximum-minimum)

    return out

