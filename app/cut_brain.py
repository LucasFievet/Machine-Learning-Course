import numpy as np

def cut_brain(sample, pos):
    data = [] 
    for d in sample:
        if pos == "lt":
            z_s,z_e = 60,d.shape[2]
            y_s,y_e = 0,60
        elif pos == "mt":
            z_s,z_e = 60,d.shape[2]
            y_s,y_e = 60,110
        elif pos == "rt":
            z_s,z_e = 50,d.shape[2]
            y_s,y_e = 110,d.shape[1]
        elif pos == "lb":
            z_s,z_e = 0,60 
            y_s,y_e = 0,60
        elif pos == "mb":
            z_s,z_e = 0,60
            y_s,y_e = 60,110
        elif pos == "rb":
            z_s,z_e = 0,50
            y_s,y_e = 110,d.shape[1]

        area = np.array([[z_s,z_e],[y_s,y_e]])

        z = range(area[0][0],area[0][1])
        y = range(area[1][0],area[1][1])
        x_len = len(d[:,0,0])
        y_len = len(y)
        z_len = len(z)

        d_new = np.zeros((x_len,y_len,z_len))
        for k in range(0,x_len):
            z_index = 0
            for i in z:
                y_index = 0
                for j in y:
                    d_new[k,y_index,z_index] = d[k,j,i]
                    y_index += 1
                z_index += 1
        data.append(d_new)
    return data
