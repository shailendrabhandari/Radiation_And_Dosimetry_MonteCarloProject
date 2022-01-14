import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from statistics import stdev

#-------------------------------------------- Task 7, 8 2 MeV -------------------------------------------

NumPhot = 20000

Einit = 2 
n=0

thetas = []

for i in range(NumPhot):
	x_tr = np.random.uniform(0,1,1)
	y_tr = np.random.uniform(0,1,1)
	theta_tr = x_tr*180
	Escat = Einit/(1+Einit/0.511*(1-np.cos(theta_tr*math.pi/180)))
	xs_tr = math.pi*(2.818e-13)**2*(Escat/Einit)**2*(Einit/Escat+Escat/Einit-np.sin(theta_tr*math.pi/180)**2)*np.sin(theta_tr*math.pi/180)/1.089378300780653e-25
	if(y_tr<=xs_tr):
		n=n+1
		thetas.append(180*x_tr)
print(n)
n, bins, patches = plt.hist(np.ravel(thetas), bins = 100)
plt.show()
print(np.max(n))

plt.hist(np.ravel(thetas), bins = 100, facecolor='blue', alpha=0.75,label='Photon angular distribution')
plt.plot(theta,(np.max(n)-12)*dsigma_dtheta/max(dsigma_dtheta),'--',color="yellow",label='Theoretical differential scattering cross section')
plt.title("Distribution of 2 MeV Photons")
plt.legend()
plt.grid()
plt.show()
