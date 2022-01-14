import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

#---------------------------------- Task 1 -------------------------------------

np.random.seed(2020)

uniform = np.random.uniform(0,1,1000)
#print(uniform)

gaussian = np.random.normal(0, 1, 1000)
print(gaussian)

x_un = np.arange(0, 1.1, 0.1)
y_un = 100*np.ones(11)

x = np.arange(-3.5, 3.5, 0.1)
y = 620.21274*norm.pdf(x, 0, 1.0)

#n, bins, patches = plt.hist(gaussian, bins = 10)
#print(bins)
#print(n)

plt.hist(uniform, bins = 12, facecolor='yellow', alpha=0.75,label='Generated data')
plt.plot(x_un,y_un,'b--',label='Predicted value')
plt.xlabel('Bin number')
plt.ylabel('Number of counts')
plt.title('Random number from the uniform distribution')
plt.grid(True)
plt.legend()
plt.show()


plt.hist(gaussian, bins = 16, facecolor='green', alpha=0.75,label='Generated data')
plt.plot(x,y,'r--',label='Predicted value')
plt.xlabel('Bin number')
plt.ylabel('Number of counts')
plt.title('Random number from the Gaussian distribution')
#plt.grid(True)
plt.legend()
plt.show()
