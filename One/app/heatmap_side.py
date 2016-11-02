"""Description of this file."""


import matplotlib.pyplot as plt

import numpy as np

from .load_data_3d import load_sample_input

def heatmap_side():
    sample = 20
    data = load_sample_input(sample)

    length = len(data[:, 1, 1])
    heat_map = data[0, :, :]

    for i in range(0, length):
        heat_map += data[i, :, :]

    #fig = plt.figure()
    #ax = fig.gca()
    plt.xticks(np.arange(0,len(data[0, 0, :]),10))
    plt.yticks(np.arange(0,len(data[0, :, 0]),10))
    plt.clf()
    plt.imshow(heat_map)
    plt.grid()
    plt.savefig("plots/avg_side_view.pdf")

    cut_brain(data)

def cut_brain(d):
    data = cut_frontal_lobe(d)
    heat_map = data[0][90,:,:]
    plt.clf()
    plt.imshow(heat_map)
    plt.grid()
    plt.savefig("plots/frontal_lobe.pdf")

def cut_frontal_lobe(sample):
    frontal_lobe = np.array([[60,130],[150,100]])

    data = [] 
    data_3d = sample
    z = range(frontal_lobe[0][0],len(data_3d[0,0,:]))
    y = range(frontal_lobe[0][1],len(data_3d[0,:,0]))
    x_len = len(data_3d[:,0,0])
    y_len = len(y)
    z_len = len(z)
    # data_fl = np.zeros(data_3d.shape)
    data_fl = np.zeros((x_len,y_len,z_len))
    for k in range(0,x_len):
        z_index = 0
        for i in z:
            y_index = 0
            for j in y:
                data_fl[k,y_index,z_index] = data_3d[k,j,i]
                y_index += 1
            z_index += 1
    data.append(data_fl)
    return data
