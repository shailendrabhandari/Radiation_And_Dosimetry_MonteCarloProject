import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from statistics import stdev
#---------------------------------- Task 4 -------------------------------------

nv = 3.43e23

Emax = 2000
Emin = 200

x = np.logspace(-2, 3, 100)

alpha = x/0.511

sigma_e = 2*math.pi*(2.818e-13)**2*((1+alpha)/(alpha**2)*(2*(1+alpha)/(1+2*alpha)-np.log(1+2*alpha)/alpha)+np.log(1+2*alpha)/alpha/2-(1+3*alpha)/((1+2*alpha)**2))

sigma_O_mass = 6.022e23*8/16*sigma_e
sigma_H_mass = 6.022e23*1/1*sigma_e

sigma_tot = 2/18*sigma_H_mass+16/18*sigma_O_mass


plt.plot(x,sigma_e,'y--',label='Electronic cross-section')
plt.xlabel('Energy, MeV')
plt.ylabel('$\sigma_e$, cm$^2$')
plt.title('Electronic Cross-section for Compton Scattering')
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.xlim(0.01,500)
plt.ylim(1e-27,1e-24)
plt.legend()
plt.show()

plt.plot(x,sigma_tot,'y--',label='Attenuation coefficient $\mu$')
plt.xlabel('Energy, MeV')
plt.ylabel('$\mu$, cm$^{-1}$')
plt.title('Attenuation coefficient, Compton scattering, H$_2$O')
plt.xscale("log")
plt.yscale("log")
plt.plot([2,2],[0.0001,0.049],"--",color="cadetblue", linewidth=1)
plt.plot([0.0001,2],[0.049,0.049],"--",color="cadetblue", linewidth=1)
plt.text(2.2, 0.002, r'2 MeV photons', fontsize=13,color="cadetblue")
plt.plot([0.2,0.2],[0.0001,0.137],"--",color="indigo", linewidth=1)
plt.plot([0.0001,0.2],[0.137,0.137],"--",color="indigo", linewidth=1)
plt.text(0.22, 0.004, r'200 keV photons', fontsize=13,color="indigo")
plt.grid(True)
plt.xlim(0.01,500)
plt.ylim(0.0001,0.3)
plt.legend()
plt.show()
