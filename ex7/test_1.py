import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from statistics import stdev
#---------------------------------- Task 4 -------------------------------------

nv = 3.43e23

Emax = 2000
Emin = 50

x = np.logspace(-2, 3, 100)

alpha = x/0.511

sigma_e = 2*math.pi*(2.818e-13)**2*((1+alpha)/(alpha**2)*(2*(1+alpha)/(1+2*alpha)-np.log(1+2*alpha)/alpha)+np.log(1+2*alpha)/alpha/2-(1+3*alpha)/((1+2*alpha)**2))

sigma_O_mass = 6.022e23*8/16*sigma_e
sigma_H_mass = 6.022e23*1/1*sigma_e

sigma_tot = 2/18*sigma_H_mass+16/18*sigma_O_mass


plt.plot(x,sigma_e,'r--',label='Electronic cross-section')
plt.xlabel('Energy, MeV')
plt.ylabel('$\sigma_e$, cm$^2$')
plt.title('Electronic cross section for Compton scattering')
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.xlim(0.01,500)
plt.ylim(1e-27,1e-24)
plt.legend()
plt.show()

plt.plot(x,sigma_tot,'r--',label='Attenuation coefficient $\mu$')
plt.xlabel('Energy, MeV')
plt.ylabel('$\mu$, cm$^{-1}$')
plt.title('Attenuation coefficient, Compton scattering, H$_2$O')
plt.xscale("log")
plt.yscale("log")
plt.plot([2,2],[0.0001,0.049],"--",color="cadetblue", linewidth=1)
plt.plot([0.0001,2],[0.049,0.049],"--",color="cadetblue", linewidth=1)
plt.text(2.2, 0.002, r'2 MeV photons', fontsize=13,color="cadetblue")
plt.plot([0.2,0.2],[0.0001,0.137],"--",color="indigo", linewidth=1)
plt.plot([0.0001,0.2],[0.137,0.137],"--",color="indigo", linewidth=1)
plt.text(0.22, 0.004, r'200 keV photons', fontsize=13,color="indigo")
plt.grid(True)
plt.xlim(0.01,500)
plt.ylim(0.0001,0.3)
plt.legend()
plt.show()

alpha = 2/0.511

sigma_e_2 = 2*math.pi*(2.818e-13)**2*((1+alpha)/(alpha**2)*(2*(1+alpha)/(1+2*alpha)-np.log(1+2*alpha)/alpha)+np.log(1+2*alpha)/alpha/2-(1+3*alpha)/((1+2*alpha)**2))

sigma_O_mass_2 = 6.022e23*8/16*sigma_e_2
sigma_H_mass_2 = 6.022e23*1/1*sigma_e_2

sigma_tot_2 = 2/18*sigma_H_mass_2+16/18*sigma_O_mass_2

print(sigma_tot_2)

#----------------------------------------------- Task 5 --------------------------------------------

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

#----------------------------------------------- Task 6 --------------------------------------------

NumSim = 1000
Pathlengths = np.zeros(NumSim)

for i in range(NumSim):
	ksi = np.random.uniform(0,1,1)
	Pathlengths[i] = - np.log(1-ksi)/sigma_tot_2

print(np.mean(Pathlengths))
print(1/sigma_tot_2)
print(stdev(Pathlengths))

x_1 = np.arange(170)
y_1 = 74.61415*np.exp(-x_1/20.421772096725537)

plt.hist(Pathlengths, bins = 100, facecolor='lightseagreen', alpha=0.75,label='Photon distribution')
plt.plot(x_1,y_1,'r--',label='Predicted value')
plt.xlabel('Pathlength, cm')
plt.ylabel('Number of photons')
plt.title('Distribution of the pathlengths for 2 MeV photons')
plt.grid(True)
plt.xlim(0,140)
plt.legend()
plt.show()

Pathlengths = np.zeros(NumSim)

for i in range(NumSim):
	ksi = np.random.uniform(0,1,1)
	Pathlengths[i] = - np.log(1-ksi)/sigma_tot_3

print(np.mean(Pathlengths))
print(1/sigma_tot_3)
print(stdev(Pathlengths))

x_1 = np.arange(46)
y_1 = 55.17867*np.exp(-x_1/7.662871195243166)

plt.hist(Pathlengths, bins = 100, facecolor='lightseagreen', alpha=0.75,label='Photon distribution')
plt.plot(x_1,y_1,'r--',label='Predicted value')
plt.xlabel('Pathlength, cm')
plt.ylabel('Number of photons')
plt.title('Distribution of the pathlengths for 200 keV photons')
plt.grid(True)
plt.xlim(0,50)
plt.legend()
plt.show()




 112  MC_3.py 
@@ -0,0 +1,112 @@
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from statistics import stdev

#-------------------------------------------- Task 7 -------------------------------------------
theta = np.arange(180)

Einit = 2 

Escat = Einit/(1+Einit/0.511*(1-np.cos(theta*math.pi/180)))

dsigma_dtheta = math.pi*(2.818e-13)**2*(Escat/Einit)**2*(Einit/Escat+Escat/Einit-np.sin(theta*math.pi/180)**2)*np.sin(theta*math.pi/180)

plt.plot(theta/180,dsigma_dtheta/max(dsigma_dtheta),'--',color="darkcyan",label='Normalization of Compton cross section')
plt.fill_between(theta/180,dsigma_dtheta/max(dsigma_dtheta), color='lightseagreen', alpha=0.2, hatch='/')
plt.xlabel('$\u03B8$, normalized')
plt.ylabel('d$\sigma$/d\u03B8, normalized')
plt.title('Compton differential cross section, 2 MeV photons')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

print(max(dsigma_dtheta))


#-------------------------------------------- Task 7, 8 2 MeV -------------------------------------------

NumPhot = 20000

Einit = 2 
n=0

thetas = []

for i in range(NumPhot):
	x_tr = np.random.uniform(0,1,1)
	y_tr = np.random.uniform(0,1,1)
	theta_tr = x_tr*180
	Escat = Einit/(1+Einit/0.511*(1-np.cos(theta_tr*math.pi/180)))
	xs_tr = math.pi*(2.818e-13)**2*(Escat/Einit)**2*(Einit/Escat+Escat/Einit-np.sin(theta_tr*math.pi/180)**2)*np.sin(theta_tr*math.pi/180)/1.089378300780653e-25
	if(y_tr<=xs_tr):
		n=n+1
		thetas.append(180*x_tr)
print(n)
n, bins, patches = plt.hist(np.ravel(thetas), bins = 100)
plt.show()
print(np.max(n))

plt.hist(np.ravel(thetas), bins = 100, facecolor='lightseagreen', alpha=0.75,label='Photon angular distribution')
plt.plot(theta,(np.max(n)-12)*dsigma_dtheta/max(dsigma_dtheta),'--',color="darkcyan",label='Theoretical differential scattering cross section')
plt.title("Distribution of 2 MeV photons")
plt.legend()
plt.grid()
plt.show()



#-------------------------------------------- Task 7, 8 200 keV -------------------------------------------


Einit = 0.2 

Escat = Einit/(1+Einit/0.511*(1-np.cos(theta*math.pi/180)))

dsigma_dtheta = math.pi*(2.818e-13)**2*(Escat/Einit)**2*(Einit/Escat+Escat/Einit-np.sin(theta*math.pi/180)**2)*np.sin(theta*math.pi/180)

plt.plot(theta/180,dsigma_dtheta/max(dsigma_dtheta),'--',color="darkcyan",label='Normalization of Compton cross section')
plt.fill_between(theta/180,dsigma_dtheta/max(dsigma_dtheta), color='lightseagreen', alpha=0.2, hatch='/')
plt.xlabel('$\u03B8$, normalized')
plt.ylabel('d$\sigma$/d\u03B8, normalized')
plt.title('Compton differential cross section, 200 keV photons')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

print(max(dsigma_dtheta))




NumPhot = 20000

Einit = 0.2 
n=0

thetas = []

for i in range(NumPhot):
	x_tr = np.random.uniform(0,1,1)
	y_tr = np.random.uniform(0,1,1)
	theta_tr = x_tr*180
	Escat = Einit/(1+Einit/0.511*(1-np.cos(theta_tr*math.pi/180)))
	xs_tr = math.pi*(2.818e-13)**2*(Escat/Einit)**2*(Einit/Escat+Escat/Einit-np.sin(theta_tr*math.pi/180)**2)*np.sin(theta_tr*math.pi/180)/2.152145827769177e-25
	if(y_tr<=xs_tr):
		n=n+1
		thetas.append(180*x_tr)
print(n)
n, bins, patches = plt.hist(np.ravel(thetas), bins = 100)
plt.show()
print(np.max(n))

plt.hist(np.ravel(thetas), bins = 100, facecolor='lightseagreen', alpha=0.75,label='Photon angular distribution')
plt.plot(theta,(np.max(n)-25)*dsigma_dtheta/max(dsigma_dtheta),'--',color="darkcyan",label='Theoretical differential \nscattering cross section')
plt.title("Distribution of 200 keV photons")
plt.legend()
plt.grid()
plt.show()



 226  MC_4.py 
@@ -0,0 +1,226 @@
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

"""
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
plt.plot(x[NumPart-1], y[NumPart-1], linewidth=1,label='Nth photon')
plt.xlabel('x, cm')
plt.ylabel('y, cm')
plt.title('Trajectories of 2 MeV photons')
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
"""
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


plt.plot(dist_mean, mean_eng, color = "teal")
plt.xlabel('distance$_{mean}$, cm')
plt.ylabel('E$_{mean}$, cm')
plt.title('Mean energy vs mean distance for 2 MeV photons')
plt.grid(True)
plt.show()

