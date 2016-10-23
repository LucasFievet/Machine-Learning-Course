import numpy as np

def normalize(inputs, minmax=None):
    data = np.array(inputs)
    if minmax == None:
        minimum = data.min()
        maximum = data.max()
        return (data-minimum)/(maximum-minimum), [minimum,maximum]
    else:
        minimum = minmax[0]
        maximum = minmax[1]
        return (data-minimum)/(maximum-minimum)

