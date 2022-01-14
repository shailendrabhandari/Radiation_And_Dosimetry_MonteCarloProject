
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

def PDF(x, y): # Function of distance x and attenuation coefficient mu (y here)
    return y*np.exp(-x*y)

def CPD(x, y):
    return 1-np.exp(-x*y)

x = np.arange(50)
hn=200 # photon energy 200 keV in this example
plt.plot(x,PDF(x, mu(hn)))
hn=2000
plt.plot(x,PDF(x, mu(hn)))


plt.figure()
hn=2000
plt.plot(x,CPD(x, mu(hn)))
hn=2000
plt.plot(x,CPD(x, mu(hn)))
plt.show()
