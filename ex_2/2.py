
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

NumParticles = 1000
NumSteps = 100
Steps = np.arange(NumSteps)
Positions = np.zeros((NumParticles,NumSteps))

for i in range(NumParticles):
	for j in range(1,NumSteps):
		step = np.random.normal(0, 1, 1)
		Positions[i, j]=Positions[i, j-1]+step
print(len(Positions[0,:]))

x = np.arange(100)
y = np.zeros(100)

plt.plot(Steps,Positions[0,:],color="paleturquoise",linewidth=0.5)
plt.plot(Steps,Positions[99,:],color="lightseagreen",linewidth=0.5)
plt.plot(Steps,Positions[199,:],color="cadetblue",linewidth=0.5)
plt.plot(Steps,Positions[299,:],color="deepskyblue",linewidth=0.5)
plt.plot(Steps,Positions[399,:],color="lawngreen",linewidth=0.5)
plt.plot(Steps,Positions[499,:],color="lightpink",linewidth=0.5)
plt.plot(Steps,Positions[599,:],color="navy",linewidth=0.5)
plt.plot(Steps,Positions[699,:],color="indigo",linewidth=0.5)
plt.plot(Steps,Positions[799,:],color="red",linewidth=0.5)
plt.plot(Steps,Positions[899,:],color="darkorange",linewidth=0.5)
plt.plot(Steps,Positions[999,:],color="coral",linewidth=0.5,label='Nth particle')
plt.plot(x,y,'b--',label='Mean value')
plt.xlabel('Collision number')
plt.ylabel('x, position')
plt.title('1-D Trajectories for 10 Particles')
plt.legend()
plt.show()

