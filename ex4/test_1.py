
import numpy as np
import matplotlib.pyplot as plt
def mu(x):  # This function returns mu depending on energy x
    nv=3.43e23
    k=x/511.0
    t1 = (1+k) * ( (2*(1+k)/(1+2*k)) - (np.log(1+2*k)/k) ) / (k**2)
    t2 = np.log(1+2*k)/(2*k)
    t3 = (1+3*k)/((1+2*k)**2)
    t=t1+t2-t3
    sig=2*np.pi*((2.81179e-13)**2)*t
    return nv*sig

#x = np.arange(20)
hn=np.linspace(50,2000,100) # in keV
fig = plt.figure()
plt.plot(hn, mu(hn))
plt.xlabel("Photon Energy(keV)")
plt.ylabel("Attenuation coefficient(μ)")
print (mu(hn))
plt.show()

