import numpy as np

def squared_error(ys, yp):
    se = sum((ys-yp)**2)/len(ys)
    return se
