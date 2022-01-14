import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from statistics import stdev

#-------------------------------------------- Task 9 -------------------------------------------

Einit = 2

NumPart = 1000

Ethr = 0.05

x = []
y = []
z = []
eng = []
thetas = []
phis = []

x_i = []
y_i = []
z_i = []
eng_i = []
theta_i = []
phi_i = []

x_cur = 0
y_cur = 0
z_cur = 0
e_cur = Einit
theta_cur = 0
phi_cur = 0
parthlength = 0

x_s = 0
y_s = 0
z_s = 0
e_s = Einit
theta_s = 0
phi_s = 0

for i in range(NumPart):
	# Setting initial positions
	x_cur = np.random.uniform(0,10,1)
	y_cur = np.random.uniform(0,10,1)
	z_cur = 0
	e_cur = Einit
	theta_cur = 90
	phi_cur = 0

	x_i.append(x_cur)
	y_i.append(y_cur)
	z_i.append(z_cur)
	theta_i.append(theta_cur)
	phi_i.append(phi_cur)
	eng_i.append(e_cur)

	while(e_cur>0.05):
		# Finding mu and finding position
		alpha = e_cur/0.511
		mu_mass = 2*math.pi*(2.818e-13)**2*((1+alpha)/(alpha**2)*(2*(1+alpha)/(1+2*alpha)-np.log(1+2*alpha)/alpha)+np.log(1+2*alpha)/alpha/2-(1+3*alpha)/((1+2*alpha)**2))
		mu_O_mass_2 = 6.022e23*8/16*mu_mass
		mu_H_mass_2 = 6.022e23*1/1*mu_mass
		mu = 2/18*mu_H_mass_2+16/18*mu_O_mass_2

		# Sampling pathlength
		ksi = np.random.uniform(0,1,1)
		pathlength = - np.log(1-ksi)/mu

		# Sampling theta
		x_tr = np.random.uniform(0,1,1)
		y_tr = np.random.uniform(0,1,1)
		x_tr = -1

		while(y_tr>x_tr):
			x_tr = np.random.uniform(0,1,1)
			y_tr = np.random.uniform(0,1,1)
			theta = np.arange(180)
			Escatt = e_cur/(1+e_cur/0.511*(1-np.cos(theta*math.pi/180)))
			dsigma_dtheta = math.pi*(2.818e-13)**2*(Escatt/Einit)**2*(Einit/Escatt+Escatt/Einit-np.sin(theta*math.pi/180)**2)*np.sin(theta*math.pi/180)

			theta_tr = x_tr*180
			Escat = e_cur/(1+e_cur/0.511*(1-np.cos(theta_tr*math.pi/180)))
			dsigma_dtheta = math.pi*(2.818e-13)**2*(Escat/Einit)**2*(Einit/Escat+Escat/Einit-np.sin(theta*math.pi/180)**2)*np.sin(theta*math.pi/180)
			xs_tr = math.pi*(2.818e-13)**2*(Escat/Einit)**2*(Einit/Escat+Escat/Einit-np.sin(theta_tr*math.pi/180)**2)*np.sin(theta_tr*math.pi/180)/max(dsigma_dtheta)
		theta_i.append(180*x_tr)
		theta_s = 180*x_tr 

		#Setting new coordinates
		ksi = np.random.uniform(0,1,1)
		phi_s = 360*ksi

		cos_theta_n_p_1 = np.sin(theta_cur*math.pi/180)*np.sin(theta_s*math.pi/180)*np.cos(phi_s*math.pi/180)+np.cos(theta_s*math.pi/180)*np.cos(theta_cur*math.pi/180)
		theta_n_p_1 = 180/math.pi*np.arccos(cos_theta_n_p_1)

		x_s = x_cur+pathlength*np.sin(theta_n_p_1*math.pi/180)*np.cos(phi_s*math.pi/180)
		y_s = y_cur+pathlength*np.sin(theta_n_p_1*math.pi/180)*np.sin(phi_s*math.pi/180)
		z_s = z_cur+pathlength*np.cos(theta_s*math.pi/180)
		e_s = e_cur/(1+e_cur/0.511*(1-np.cos(theta_n_p_1*math.pi/180)))
		theta_s = theta_n_p_1

		theta_i.append(theta_s)
		phi_i.append(phi_s)
		eng_i.append(e_s)
		x_i.append(x_s)
		y_i.append(y_s)
		z_i.append(z_s)

		x_cur = x_s
		y_cur = y_s
		z_cur = z_s
		e_cur = e_s
		theta_cur = theta_s
		phi_cur = phi_s

	x_i_rav=np.ravel(x_i)
	y_i_rav=np.ravel(y_i)
	z_i_rav = np.ravel(z_i)
	theta_i_rav = np.ravel(theta_i)
	phi_i_rav = np.ravel(phi_i)
	eng_i_rav = np.ravel(eng_i)

	x.append(x_i_rav)
	y.append(y_i_rav)
	z.append(z_i_rav)
	thetas.append(theta_i_rav)
	phis.append(phi_i_rav)
	eng.append(eng_i_rav)

	x_i =[]
	y_i =[]
	z_i =[]
	theta_i = []
	phi_i = []
	eng_i = []

# Rewriting arrays


for i in range(NumPart-1):
	x_len = len(x[i])
	z_new = np.zeros(x_len)
	for j in range(x_len):
		z_new[j]=-z[i][j]
	e_new = np.zeros(x_len)
	for j in range(x_len):
		e_new[j]=eng[i][j]
	#plt.plot(z_new, x[i], linewidth=1)
	plt.plot(x[i], y[i], linewidth=1)
x_len = len(x[NumPart-1])
z_new = np.zeros(x_len)
for j in range(x_len):
	z_new[j]=-z[NumPart-1][j]
e_new = np.zeros(x_len)
for j in range(x_len):
	e_new[j]=eng[NumPart-1][j]
#plt.plot(z_new, x[NumPart-1], linewidth=1,label='Nth photon')
plt.plot(x[NumPart-1], y[NumPart-1], linewidth=1,label='Nth Photon')
plt.xlabel('x, cm')
plt.ylabel('y, cm')
plt.title('Trajectories of 200 keV Photons')
x1 = np.arange(0, 10, 1)
y1 = np.zeros(10)
y2 = 10*np.ones(10)
y3 = np.arange(0, 10, 1)
x2 = np.zeros(10)
x3 = 10*np.ones(10)
plt.plot(x1,y1,'b--', linewidth=3 ,label='Generation surface')
plt.plot(x1,y2,'b--', linewidth=3 )
plt.plot(x2,y3,'b--', linewidth=3 )
plt.plot(x3,y3,'b--', linewidth=3 )
plt.legend()
plt.show()

lengths =np.zeros(NumPart)
for i in range(NumPart):
	x_len = len(x[i])
	z_new = np.zeros(x_len)
	for j in range(x_len):
		z_new[j]=-z[i][j]
	lengths[i]=len(z_new)

print(np.max(lengths))

energies = np.zeros((NumPart,np.int(np.max(lengths))))
zs = np.zeros((NumPart,np.int(np.max(lengths))))
dist = np.zeros((NumPart,np.int(np.max(lengths))))

for i in range(NumPart):
	x_len = len(x[i])
	z_len = len(z[i])
	zs[i,:]=np.ones(np.int(np.max(lengths)))*z[i][x_len-1]
	dist[i,:]=np.ones(np.int(np.max(lengths)))*np.sqrt(z[i][x_len-1]**2+x[i][x_len-1]**2+y[i][x_len-1]**2)

for i in range(NumPart):
	x_len = len(x[i])
	for j in range(x_len):
		energies[i][j]=eng[i][j]
		dist[i][j]=np.sqrt((z[i][j])**2+(x[i][j])**2+(y[i][j])**2)

mean_eng = np.zeros(np.int(np.max(lengths)))
dist_mean = np.zeros(np.int(np.max(lengths)))

for i in range(np.int(np.max(lengths))):
	mean_eng[i]=np.mean(energies[:,i])
	dist_mean[i]=np.mean(dist[:,i])


plt.plot(dist_mean, mean_eng, color = "blue")
plt.xlabel('distance$_{mean}$, cm')
plt.ylabel('E$_{mean}$, cm')
plt.title('Mean energy vs mean distance for 2 MeV hotons')
plt.grid(True)
plt.show()
