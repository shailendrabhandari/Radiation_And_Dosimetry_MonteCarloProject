import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from statistics import stdev

NumSim = 1000
Pathlengths = np.zeros(NumSim)

for i in range(NumSim):
	ksi = np.random.uniform(0,1,1)
	Pathlengths[i] = - np.log(1-ksi)/sigma_tot_2

print(np.mean(Pathlengths))
print(1/sigma_tot_2)
print(stdev(Pathlengths))

x_1 = np.arange(170)
y_1 = 74.61415*np.exp(-x_1/20.421772096725537)

plt.hist(Pathlengths, bins = 100, facecolor='lightseagreen', alpha=0.75,label='Photon distribution')
plt.plot(x_1,y_1,'r--',label='Predicted value')
plt.xlabel('Pathlength, cm')
plt.ylabel('Number of photons')
plt.title('Distribution of the pathlengths for 2 MeV photons')
plt.grid(True)
plt.xlim(0,140)
plt.legend()
plt.show()

Pathlengths = np.zeros(NumSim)

for i in range(NumSim):
	ksi = np.random.uniform(0,1,1)
	Pathlengths[i] = - np.log(1-ksi)/sigma_tot_3

print(np.mean(Pathlengths))
print(1/sigma_tot_3)
print(stdev(Pathlengths))

x_1 = np.arange(46)
y_1 = 55.17867*np.exp(-x_1/7.662871195243166)

plt.hist(Pathlengths, bins = 100, facecolor='lightseagreen', alpha=0.75,label='Photon distribution')
plt.plot(x_1,y_1,'r--',label='Predicted value')
plt.xlabel('Pathlength, cm')
plt.ylabel('Number of photons')
plt.title('Distribution of the pathlengths for 200 keV photons')
plt.grid(True)
plt.xlim(0,50)
plt.legend()
plt.show()

