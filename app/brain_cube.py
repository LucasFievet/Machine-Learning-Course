from .load_deviations import load_deviation
from .slice3 import slice3, meshgrid3
import numpy as np

def brain_cube():
    ID = 10
    data = load_deviation(ID)
    # Number of x-grid points
    nx = len(data[:][0][0])

    # Number of 
    ny = len(data[0][:][0])
    nz = len(data[0][0][:])

    x = np.linspace(0,nx,nx)
    y = np.linspace(0,ny,ny)
    z = np.linspace(0,nz,nz)

    xx,yy,zz = meshgrid3(x,y,z)
     
    # Display three cross sections of a Gaussian Beam/Paraxial wave
    #u = np.real(np.exp(-(2*xx**2+yy**2)/(.2+2j*zz))/np.sqrt(.2+2j*zz))
    s3 = slice3(xx,yy,zz,data)
    s3.xlabel('x',fontsize=18)
    s3.ylabel('y',fontsize=18)
    s3.zlabel('z',fontsize=18)
     

    s3.show()
