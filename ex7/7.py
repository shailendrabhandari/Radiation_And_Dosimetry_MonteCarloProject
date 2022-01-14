import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from statistics import stdev

#-------------------------------------------- Task 7 -------------------------------------------
theta = np.arange(180)

Einit = 0.2

Escat = Einit/(1+Einit/0.511*(1-np.cos(theta*math.pi/180)))

dsigma_dtheta = math.pi*(2.818e-13)**2*(Escat/Einit)**2*(Einit/Escat+Escat/Einit-np.sin(theta*math.pi/180)**2)*np.sin(theta*math.pi/180)

plt.plot(theta/180,dsigma_dtheta/max(dsigma_dtheta),'--',color="blue",label='Normalization of Compton cross section')
plt.fill_between(theta/180,dsigma_dtheta/max(dsigma_dtheta), color='yellow', alpha=0.2, hatch='/')
plt.xlabel('$\u03B8$, normalized')
plt.ylabel('d$\sigma$/d\u03B8, normalized')
plt.title('Compton Differential Cross-section, 200 keV Photons')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

print(max(dsigma_dtheta))

