import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from statistics import stdev

#-------------------------------------------- Task 7, 8 200 keV -------------------------------------------


Einit = 0.2 

Escat = Einit/(1+Einit/0.511*(1-np.cos(theta*math.pi/180)))

dsigma_dtheta = math.pi*(2.818e-13)**2*(Escat/Einit)**2*(Einit/Escat+Escat/Einit-np.sin(theta*math.pi/180)**2)*np.sin(theta*math.pi/180)

plt.plot(theta/180,dsigma_dtheta/max(dsigma_dtheta),'--',color="darkcyan",label='Normalization of Compton cross section')
plt.fill_between(theta/180,dsigma_dtheta/max(dsigma_dtheta), color='lightseagreen', alpha=0.2, hatch='/')
plt.xlabel('$\u03B8$, normalized')
plt.ylabel('d$\sigma$/d\u03B8, normalized')
plt.title('Compton differential cross section, 200 keV photons')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

print(max(dsigma_dtheta))




NumPhot = 20000

Einit = 0.2 
n=0

thetas = []

for i in range(NumPhot):
	x_tr = np.random.uniform(0,1,1)
	y_tr = np.random.uniform(0,1,1)
	theta_tr = x_tr*180
	Escat = Einit/(1+Einit/0.511*(1-np.cos(theta_tr*math.pi/180)))
	xs_tr = math.pi*(2.818e-13)**2*(Escat/Einit)**2*(Einit/Escat+Escat/Einit-np.sin(theta_tr*math.pi/180)**2)*np.sin(theta_tr*math.pi/180)/2.152145827769177e-25
	if(y_tr<=xs_tr):
		n=n+1
		thetas.append(180*x_tr)
print(n)
n, bins, patches = plt.hist(np.ravel(thetas), bins = 100)
plt.show()
print(np.max(n))

plt.hist(np.ravel(thetas), bins = 100, facecolor='blue', alpha=0.75,label='Photon angular distribution')
plt.plot(theta,(np.max(n)-25)*dsigma_dtheta/max(dsigma_dtheta),'--',color="yellow",label='Theoretical differential \nscattering cross section')
plt.title("Distribution of 200 keV photons")
plt.legend()
plt.grid()
plt.show()

