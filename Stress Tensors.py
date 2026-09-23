import numpy as np
import matplotlib.pyplot as plt

hst_file = "64 Resolution.hst"


data = np.loadtxt(hst_file, comments="#")

time       = data[:,0]
maxwell    = data[:,13]
reynolds   = data[:,14]

if np.mean(data[:,9])<10e-02 or np.mean(data[:,9])>10e+10:
    Omega0,p0 = 1e-3,1e-6
else:
    Omega0,p0 = 1.0,01e-1


maxwell = maxwell /(4*np.pi)
alpha = (reynolds + maxwell) / p0
orbits = Omega0 * time / (2*np.pi)
alpha_avg  = np.mean(alpha)


print("Average alpha:", alpha_avg)

plt.figure(figsize=(10,6))

plt.plot(orbits, maxwell/p0, label="Maxwell Parameter", color="red")
plt.plot(orbits, reynolds/p0, label="Reynolds Parameter", color="green")
plt.plot(orbits, alpha, label="Alpha Parameter", color="blue")
plt.xlabel("Orbits")

plt.ylim(10e-3,10)
plt.ylabel("Stress")
plt.title("MRI Stress vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
