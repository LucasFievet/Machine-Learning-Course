import numpy as np

from .load_data import load_sample_input

def cut_frontal_lobe(sample):
    frontal_lobe = np.array([[130,60],[100,150]])

    data = load_sample_input(sample)
    data_3d = data.get_data()[:,:,:,0]

    x = range(60,len(data_3d[0,0,:]))
    y = range(130,len(data_3d[0, :, 0]))
    
    data_fl = np.zeros(data_3d.shape)
    for k in range(0,len(data_3d[:, 0, 0])):
        for i in x:
            for j in y:
                data_fl[k,j,i] += data_3d[k,j,i]
    return data_fl
