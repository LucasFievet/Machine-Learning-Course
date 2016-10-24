import numpy as np

def normalize(inputs, minmax=None):
    data = np.array(inputs)
    if minmax == None:
        minimum = data.min()
        maximum = data.max()
        return (data-minimum)/(maximum-minimum), [minimum,maximum]
    else:
        return (data-minmax[0])/(minmax[1]-minmax[0])

