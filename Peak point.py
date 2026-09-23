import numpy as np

hst_file = "HGB.hst"
data = np.loadtxt(hst_file, comments="#")
p = 0.000001

for i in range(data.shape[0]):
    if i+1==data.shape[0]:
        break
    elif (data[i+1,13]-data[i,13])<0:
        print(data[i,13]/(p*4*np.pi))
        break
    else:
        continue
