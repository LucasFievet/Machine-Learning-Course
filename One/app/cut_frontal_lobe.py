import numpy as np

def cut_frontal_lobe(sample):
    frontal_lobe = np.array([[60,130],[150,100]])

    data = [] 
    for data_3d in sample:
        print(data_3d)
        print(len(data_3d[0,0,:]))
        z = range(frontal_lobe[0][0],len(data_3d[0,0,:]))
        y = range(frontal_lobe[0][1],len(data_3d[0,:,0]))
        x_len = len(data_3d[:,0,0])
        y_len = len(y)
        z_len = len(z)
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
