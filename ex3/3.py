#Code written by Shailendra Bhandari 2020/2021 Radiation and Dosimetry Course
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math



KinEng = 100

NumParticles = 1000
NumSteps = 120
k = 1.26466
Steps = np.arange(NumSteps)
Positions = np.zeros((NumParticles,NumSteps))
Energies = np.zeros((NumParticles,NumSteps))

for i in range(NumParticles):
	Energies[i,0]=KinEng

for i in range(NumParticles):
	for j in range(1,NumSteps):
		step = np.random.normal(0, 1, 1)
		Energies[i, j] = Energies[i, j-1]-k*abs(step)

FinalEnergies = np.zeros(NumParticles)
for i in range(NumParticles):
	FinalEnergies[i]=Energies[i,NumSteps-1]
print(np.mean(FinalEnergies))


for i in range(NumParticles):
	for j in range(1,NumSteps):
		step = np.random.normal(0, 1, 1)
		if(Energies[i, j-1]>0):
			Positions[i, j] = Positions[i, j-1]+step
			Energies[i, j] = Energies[i, j-1]-k*abs(step)
			if(Energies[i, j]<0):
				Energies[i, j] = 0
		else:
			Positions[i, j] = Positions[i, j-1]
			Energies[i, j] = 0


plt.plot(Steps,Energies[0,:],color="green",linewidth=0.5)

plt.plot(Steps,Energies[0,:],color="paleturquoise",linewidth=0.5)
plt.plot(Steps,Energies[99,:],color="lightseagreen",linewidth=0.5)
plt.plot(Steps,Energies[199,:],color="cadetblue",linewidth=0.5)
plt.plot(Steps,Energies[269,:],color="deepskyblue",linewidth=0.5)
plt.plot(Steps,Energies[399,:],color="lawngreen",linewidth=0.5)
plt.plot(Steps,Energies[439,:],color="lightpink",linewidth=0.5)
plt.plot(Steps,Energies[539,:],color="navy",linewidth=0.5)
plt.plot(Steps,Energies[669,:],color="indigo",linewidth=0.5)
plt.plot(Steps,Energies[779,:],color="red",linewidth=0.5)
plt.plot(Steps,Energies[849,:],color="darkorange",linewidth=0.5)
plt.plot(Steps,Energies[999,:],color="coral",linewidth=0.5,label='Nth particle')
plt.xlabel('Collision number')
plt.ylabel('E, MeV')
plt.title('Energy Loss for 10 Particles')
plt.legend()
plt.show()

FinalPositions = np.zeros(NumParticles)
for i in range(NumParticles):
	FinalPositions[i]=Positions[i,NumSteps-1]
print(min(FinalPositions))
print(max(FinalPositions))

x = np.arange(-35, 35, 0.1)
y = 6284.33426*norm.pdf(x, -0.20881, 10.12815)

#n, bins, patches = plt.hist(FinalPositions, bins = 10)
#print(bins)
#print(n)

plt.hist(FinalPositions, bins = 14, facecolor='yellow', alpha=0.75,label='Final positions')
plt.plot(x,y,'b--',label='Predicted value')
plt.xlabel('Bin number')
plt.ylabel('Number of counts')
plt.title('Distribution of Final Positions')
plt.grid(True)
plt.legend()
plt.show()

x = np.arange(120)
y = np.zeros(120)
plt.plot(Steps,Positions[0,:],color="paleturquoise",linewidth=0.5)
plt.plot(Steps,Positions[49,:],color="lightseagreen",linewidth=0.5)
plt.plot(Steps,Positions[129,:],color="cadetblue",linewidth=0.5)
plt.plot(Steps,Positions[289,:],color="deepskyblue",linewidth=0.5)
plt.plot(Steps,Positions[319,:],color="lawngreen",linewidth=0.5)
plt.plot(Steps,Positions[469,:],color="lightpink",linewidth=0.5)
plt.plot(Steps,Positions[539,:],color="navy",linewidth=0.5)
plt.plot(Steps,Positions[699,:],color="indigo",linewidth=0.5)
plt.plot(Steps,Positions[729,:],color="red",linewidth=0.5)
plt.plot(Steps,Positions[819,:],color="darkorange",linewidth=0.5)
plt.plot(Steps,Positions[999,:],color="coral",linewidth=0.5,label='Nth particle')
plt.plot(x,y,'y--',label='Mean value')
plt.xlabel('Collision number')
plt.ylabel('x, position')
plt.title('1-D Trajectories for 10 Particles')
plt.legend()
plt.show()

plt.plot(Positions[0,:],Energies[0,:],color="paleturquoise",linewidth=0.5)
plt.plot(Positions[99,:],Energies[99,:],color="lightseagreen",linewidth=0.5)
plt.plot(Positions[199,:],Energies[199,:],color="cadetblue",linewidth=0.5)
plt.plot(Positions[299,:],Energies[299,:],color="deepskyblue",linewidth=0.5)
plt.plot(Positions[359,:],Energies[399,:],color="lawngreen",linewidth=0.5)
plt.plot(Positions[489,:],Energies[499,:],color="lightpink",linewidth=0.5)
plt.plot(Positions[599,:],Energies[599,:],color="navy",linewidth=0.5)
plt.plot(Positions[639,:],Energies[699,:],color="indigo",linewidth=0.5)
plt.plot(Positions[799,:],Energies[799,:],color="red",linewidth=0.5)
plt.plot(Positions[879,:],Energies[899,:],color="darkorange",linewidth=0.5)
plt.plot(Positions[819,:],Energies[899,:],color="coral",linewidth=0.5,label='Nth particle')
x = np.zeros(100)
y = np.arange(100)
plt.plot(x,y,'y--',label='Mean value')
plt.xlabel('x, position')
plt.ylabel('E, MeV')
plt.title('Energy versus Position of a Particles')
plt.legend()
plt.show()
