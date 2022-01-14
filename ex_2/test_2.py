# example making a random collision and showing that the distribution after a number of
# steps is a gaussian 
import numpy as np
#matplotlib inline
import matplotlib.pyplot as plt
from scipy import stats # has lots of distribution functions in it
from math import erfc   # complimentary error function 

# randomly walk nsteps and return the x value
# starting at x=0
# each step has zero mean and a variance of 1
def walkn(nsteps):  # random walk using a normal distribution for step sizes
    r = stats.norm.rvs(size=nsteps)  # normal distribution mean=0 sigma =1
    # r is a vector values randomly generated with a normal distribution
    return sum(r)  # the sum of the entire vector!

# colide npart numbers of particles  nsteps and return a vector of x positions
# the function that gives us a randomly generated position is walkn
def npart_walkn(npart,nsteps):
    xvec = np.zeros(0)
    for i in range(npart):
        x = walkn(nsteps)  # a single random collision value
        xvec = np.append(xvec,x)  # append each random collision to the vector
    return xvec

nsteps = 100 # number of steps
npart = 1000 # number of particles 
# fill a vector with npart collides each with n nsteps
xvec = npart_walkn(npart,nsteps)
# plot the histogram, i.e., measured distribution of final positions 
#   after n steps of collisions 
n, bins, patches =plt.hist(xvec,bins='auto', density=True)
# this retuns as n the number of values counted, bins is the x values of the bins, 
# patches is for plotting the histogram 
plt.xlabel("Bin number",fontsize=16)
plt.ylabel("Number of counts",fontsize=12)  # probability!

# a gaussian probability density distribution, this is a function!
mygaus = stats.norm(0.0, np.sqrt(nsteps))  # should scale with sqrt(nsteps)
y = mygaus.pdf(bins)  # evaluate the function at the bin locations
plt.plot(bins,y,"k", lw=2 )  #plot the expected density distribution as a black line
plt.show()
