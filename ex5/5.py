import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from statistics import stdev
s = np.linspace(0, 40, 100)

PDF_2 = sigma_tot_2*np.exp(-s*sigma_tot_2) 
CDF_2 = 1 - np.exp(-s*sigma_tot_2) 

plt.plot(s,PDF_2,"--",color = 'lightseagreen',label='PDF, f(s)')
plt.plot(s,CDF_2,"--",color="darkorange",label='CDF, F(s)')
plt.xlabel('s, cm')
plt.ylabel('PDF/CDF value, au')
plt.title('PDF and CDF for 2 MeV photons')
plt.grid(True)
plt.xlim(0,40)
plt.ylim(0,1)
plt.legend(loc = 'upper left')
plt.show()


alpha = 0.2/0.511

sigma_e_3 = 2*math.pi*(2.818e-13)**2*((1+alpha)/(alpha**2)*(2*(1+alpha)/(1+2*alpha)-np.log(1+2*alpha)/alpha)+np.log(1+2*alpha)/alpha/2-(1+3*alpha)/((1+2*alpha)**2))

sigma_O_mass_3 = 6.022e23*8/16*sigma_e_3
sigma_H_mass_3 = 6.022e23*1/1*sigma_e_3

sigma_tot_3 = 2/18*sigma_H_mass_3+16/18*sigma_O_mass_3

print(sigma_tot_3)


PDF_3 = sigma_tot_3*np.exp(-s*sigma_tot_3) 
CDF_3 = 1 - np.exp(-s*sigma_tot_3) 

plt.plot(s,PDF_3,"--",color = 'lightseagreen',label='PDF, f(s)')
plt.plot(s,CDF_3,"--",color="darkorange",label='CDF, F(s)')
plt.xlabel('s, cm')
plt.ylabel('PDF/CDF value, au')
plt.title('PDF and CDF for 200 keV photons')
plt.xlim(0,40)
plt.ylim(0,1)
plt.grid(True)
plt.legend(loc = 'upper left')
plt.show()

