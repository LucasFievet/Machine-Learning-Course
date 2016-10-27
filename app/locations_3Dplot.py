import os

import matplotlib.pyplot as plt
import scipy.io

from mpl_toolkits.mplot3d import Axes3D

from .settings import CURRENT_DIRECTORY

def locations_3D(w_size=5):
    cache_path = os.path.join(CURRENT_DIRECTORY ,".." ,"cache" ,"local_max_locations.mat")
    if os.path.exists(cache_path):
        out = scipy.io.loadmat(cache_path)['out']
        n = len(out)
        x , y, z = [], [], []

        print("Number of maxima =", n)

        for i in range(n):
            x.append(out[i, 0] + w_size/2)
            y.append(out[i, 1] + w_size/2)
            z.append(out[i, 2] + w_size/2)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        Axes3D.scatter(ax, xs=x, ys=y, zs=z, zdir='z')
        plt.show()
    return
