import numpy as np
import matplotlib.pyplot as plt

def squared_error(ys, yp):
    se = sum((ys-yp)**2)/len(ys)
    
    #data = np.column_stack((ys,yp)) 
    #data.view('i8,i8').sort(order=['f0'], axis=0)
    #plt.figure()
    #plt.scatter(range(0,len(data[:,0])),data[:,0],color="red")
    #plt.scatter(range(0,len(data[:,1])),data[:,1],color="blue")
    #plt.grid()
    #plt.savefig("plots/error.pdf")
    #plt.close()
    return se
