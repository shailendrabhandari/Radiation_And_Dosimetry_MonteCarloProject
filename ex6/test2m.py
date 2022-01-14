import numpy as np
import matplotlib.pyplot as plt

def mu(x):
    nv=3.43e23
    k=x/511.0
    t1 = (1+k) * ( (2*(1+k)/(1+2*k)) - (np.log(1+2*k)/k) ) / (k**2)
    t2 = np.log(1+2*k)/(2*k)
    t3 = (1+3*k)/((1+2*k)**2)
    t=t1+t2-t3
    sig=2*np.pi*((2.81179e-13)**2)*t
    return nv*sig

def iCPD(x, y):
    ran=np.random.rand(y)
    return -np.log(1-ran)/mu(x)

def PDF(x, y):
    return y*np.exp(-x*y)

hn=2000 # in keV
npart=1000 # number of photons
vec=iCPD(hn, npart)

nbin=30
hist=np.histogram(vec, bins=nbin)

yhist=hist[0]/np.sum(hist[0])
xhist=hist[1]
x1=xhist[0:nbin]
x2=xhist[1:nbin+1]
xhist=(x1+x2)/2.0
wid=(xhist[nbin-1]-xhist[0])/(1.25*nbin)
plt.figure()
plt.bar(xhist, yhist, width=wid)

y=PDF(xhist, mu(hn))/np.sum(PDF(xhist, mu(hn))) # data must be normalized to unit area
plt.plot(xhist,y, 'r-')
print(np.mean(vec))
print(1/mu(hn))
plt.show()


#For Mean and 1/mu value 200 keV 
#7.401194524856777
#7.203799074766589



#For Mean and 1/mu value 2 MeV 
#21.018238589602923
#20.007082580221535


