import numpy as np

shape = []

for a in range(2, 10):
    z = np.zeros((a*2-1,a*2-1))

    ci,cj=a-1,a-1
    cr=a-1

    I,J=np.meshgrid(np.arange(z.shape[0]),np.arange(z.shape[1]))
    dist=np.sqrt((I-ci)**2+(J-cj)**2)
    z[np.where(dist<=cr)]=1

    shape.append(z)

shapes = dict(enumerate(shape))