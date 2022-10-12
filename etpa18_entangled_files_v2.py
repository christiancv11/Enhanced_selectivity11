
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import mpmath
from mpmath import *
import scipy
from scipy import integrate
import cmath
from scipy.integrate import ode
mp.dps = 45; mp.pretty = True

##########################################################################
########################## Show initializing #############################
##########################################################################

print("==========================================================================")
print("================= Selectivity Population Dynamics in E2PA ================")
print("==========================================================================")


##########################################################################
###  Conventions for levels: ground = 0, intermediate = 1, excited = 2 ###
##########################################################################

##########################################################################
##########:::::::::::::::::::: Inputs :::::::::::::::::::::::::::#########
##########################################################################

##########################################################################
###### Number of modes and number of levels in each potential ############
##########################################################################

Nlevels = 63 #Number of levels for each potential
Modes = 200 #Number of modes for the incoming field
lex = 18 #Level of resonance

print("Number of levels for each potential: ", Nlevels)
print("Number of modes for the field: ", Modes)
print("Level of resonance: ", lex)
print("==========================================================================")

##########################################################################

##########################################################################
################## Constants for the photons profile #####################
##########################################################################

r0 = 0

#Value of the spectral width of the wave packet \sigma = \sigma _{r}^{-1} in Hz
sigma = 10e12

#Value of the spectral width of the wave packet \sigma = \sigma _{r}^{-1} in eV -> 1Hz = 4.135 58 x 10^-15 eV

sigmaev = sigma * 4.13558e-15

#Constant for entanglement

ss = 0.05
print("Constants for the photons profile : ")
print("sigma: ", sigma)
print("sigma in eV: ", sigmaev)
print("ss: ", ss)
print("==========================================================================")
##########################################################################

##########################################################################
################### Constants for the simulations ########################
##########################################################################

t0 = -150.0
t1 = 150.0
dt = 0.1

print("Simulation parameters: ")
print("t0: ", t0)
print("t1: ", t1)
print("dt: ", dt)
print("==========================================================================")

##########################################################################

##########################################################################
################### Constants for the potentials #########################
##########################################################################

#Value of the electron mass in eV
emass = 0.5110e6

#Value of the reduced mass of two nuclei in eV
mu = 19800 * emass

#Value of Bohr radius in eV^-1
a0 = 2.68e-4

##########################################################################

################# Minimum of the potential energy: epsilon ###############

#Value of epsilon_{g} in eV
epsilong = 0

#Value of epsilon_{m} in eV
epsilonm = 1.8201

#Value of epsilon_{e} in eV
epsilone = 3.7918

##########################################################################

################### Depth of the potential: D ############################

#Value of D_{g} in eV
dg = 0.7466 * 1

#Value of D_{m} in eV
dm = 1.0303 * 1

#Value of D_{e} in eV
de = 0.5718 * 1

##########################################################################

########### Range of the potential in terms of Bohr radius: a ############

ag = 2.2951

am = 3.6591

ae = 3.1226

##########################################################################

############ Equilibrium position x_{0} of the Morse Potential ###########

#Value of x_{0}^{(g)}
x0g = 5.82

#Value of x_{0}^{(m)}
x0m = 6.87

#Value of x_{0}^{(m)}
x0e = 7.08

##########################################################################

##########################################################################
################### Constants for the interactions #######################
##########################################################################

print("Constants for the interactions: ")

#### probe's constant ###
deltaalg = 1

#Definition of the relaxation rate \gamma = \gamma _{m_{\nu}} = \gamma _{e _{\nu \nu'}} in Hz
gammar = 6e6
print("relaxation rate gamma: ", gammar)

#Definition of the relaxation rate \gamma = \gamma _{m_{\nu}} = \gamma _{e _{\nu \nu'}} in eV
gammarev = gammar * 4.13558e-15
print("relaxation rate gamma in eV: ", gammarev)
print("==========================================================================")
##########################################################################

##########################################################################
############################## Functions #################################
##########################################################################

#General definition of epsilon_{\ell} in eV

def epsilonl(l):
    if l == 0:
       return epsilong
    if l == 1:
       return epsilonm
    if l == 2:
       return epsilone
    else:
       return 0 

#General definition of D_{\ell} in eV

def potdepth(l):
    if l == 0:
       return dg
    if l == 1:
       return dm
    if l == 2:
       return de
    else:
       return 0 

#General definition of $a_{\ell}$ in eV

def al(l):
    if l == 0:
       return ag * a0
    if l == 1:
       return am * a0
    if l == 2:
       return ae * a0
    else:
       return 0 

#Definition of \omega _{\ell}

def omegal(l):
    return sqrt((2*potdepth(l))/((al(l)**2) * mu))

#Definition of \chi _{\ell}

def chil(l):
    return 1/sqrt(8 * (al(l)**2) * potdepth(l) * mu)

#Definition of \omega _{\ell _{\nu}}

def omegalnu(l, nu):
    return epsilonl(l) + (omegal(l) * (nu + 0.5)) - (omegal(l) * chil(l) * (nu + 0.5)**2)


#List of energies of intermediate-potential levels
list_omegas_m = []

nus = np.array(range(25))

for i in nus:
    list_omegas_m.append(omegalnu(1, i))

list_omegas_m = np.array(list_omegas_m)
print("List of energies of intermediate-potential levels:")
print(list_omegas_m)
print("==========================================================================")
#List of energies of excited-potential levels
list_omegas_e = []

nus = np.array(range(25))

for i in nus:
    list_omegas_e.append(omegalnu(2, i))

list_omegas_e = np.array(list_omegas_e)
print("List of energies of excited-potential levels:")
print(list_omegas_e)
print("==========================================================================")

#General definition of the equilibrium position x_{0}^{(\ell)} of the Morse Potential
def x0l(l):
    if l == 0:
       return x0g * a0
    if l == 1:
       return x0m * a0
    if l == 2:
       return x0e * a0
    else:
       return 0

#Definition of j_{\ell}

def jl(l):
    return (2* al(l) * sqrt(2 * mu * potdepth(l))) - 1

#definition of y_{\ell}
def yl(x, l):
    return (jl(l) + 1)*exp(-(x - x0l(l))/al(l))

#definition of N_{j_{ell},\nu}

def nor(l, nu):
    return sqrt((fac(nu)*(jl(l) - 2*nu))/(al(l) * gamma(jl(l) - nu + 1)))

#Definition of the vibrational eigenfunction \xi

def xi(x, l, nu):
    return nor(l, nu) * exp(-yl(x, l)/2) * ((jl(l) + 1)**(0.5*jl(l) - nu)) * exp(-jl(l) * ((x - x0l(l))/(2*al(l)))) * exp(nu * ((x - x0l(l))/al(l))) * laguerre(nu, jl(l) - 2*nu, yl(x, l))

#definition of the Franck-Condon factor

def fcf(l, l1, nu, nu1):
    result = quad(lambda x: xi(x, l, nu)*xi(x, l1, nu1), [0, 10*a0])
    return np.abs(result)

#Franck-Condon Factors g->m
lista2 = []

nus = np.array(range(Nlevels))

lista2 = np.loadtxt('/home/christiandavid/Desktop/UCL_PhD/Ff_Factors_gm_63.txt')
lista2 = np.array(lista2, dtype=float)

print("Franck-Condon Factors g -> m: ")
print(lista2)
print("==========================================================================")

fig, ax1 = plt.subplots()
ax1.tick_params(direction='in', top = True, right = True, length=7)
plt.rcParams['font.size'] = '17'
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
	label.set_fontsize(17)
plt.bar(range(Nlevels), lista2)
plt.xlabel(r"$\nu$", fontsize=17)
plt.ylabel(r"$F_{\nu}$", fontsize=17)
plt.show()

lista2 = np.array(lista2, dtype=float)

#Franck-Condon Factors m->e
Fnn = np.zeros((Nlevels, Nlevels))

Fnn = np.loadtxt('/home/christiandavid/Desktop/UCL_PhD/Ff_Factors_me_63.txt')

print("Franck-Condon Factors m -> e: ")
print(Fnn)
print("==========================================================================")

fig, ax1 = plt.subplots()
ax1.tick_params(direction='in', top = True, right = True, length=7)
plt.rcParams['font.size'] = '17'
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
	label.set_fontsize(17)
img2 = ax1.imshow(Fnn,interpolation='nearest',
                    cmap = 'seismic',
                    origin='lower')
ax1.set_xlabel(r"$\nu$")
ax1.set_ylabel(r"$\nu '$")
cbar = plt.colorbar(img2,cmap='seismic')
cbar.set_label(r"$F_{\nu \nu '}$",size=17)
plt.show()

#########################################################################
## Definition of the two-photon joint amplitude for correlated pairs  ###
#########################################################################
wlength = float(4*pi/omegalnu(2, lex)) #Wavelength in ev^-1
k0 = float(2*pi / wlength) #Wave number in ev -> k_{0}
sigmaev = float(sigmaev)

print("==========================================================================")
print("=================== Parameters of the correlated pairs  ==================")
print("==========================================================================")

#Energy of incident field
print("Energy of incident field in eV: ")
print("k0: ", k0, "2k0: ", 2*k0)

#Value of \delta k in Hz
deltak = 100 * (10**9)
print("deltak: ", deltak)

#Value of \delta k in eV
deltakev = deltak * 4.13558e-15
print("delta k in eV: ", deltakev)

kz = np.arange(k0 - deltakev*(Modes/2), k0 + deltakev*(Modes/2), deltakev)
kp = np.arange(k0 - deltakev*(Modes/2), k0 + deltakev*(Modes/2), deltakev)
#kz = np.linspace(k0 - 3.6*sigmaev, k0 + 3.6*sigmaev, Modes)
#kp = np.linspace(k0 - 3.6*sigmaev, k0 + 3.6*sigmaev, Modes)

Kz, Kp = np.meshgrid(kz, kp)

print("Meshgrid: ", len(kz))
##########################################
###### Vectorization of cmath.exp ########
##########################################

def expi(x):
    return cmath.exp(-1j * x * r0)

expivec = np.vectorize(expi)

##########################################

noren = (np.pi**0.375) * (sigmaev**0.5) * ((ss*sigmaev)**0.25)

print("noren: ", noren)
print("==========================================================================")

def Psi2pen(k, k1):
    return np.exp(-((k - k0)**2)/(4 * sigmaev**2)) * (np.pi * (ss*sigmaev)**2)**(-0.25) * np.exp(-((k + k1 - 2*k0)**2)/(4*((ss*sigmaev)**2))) * expivec(k) * expivec(k1) * (1/noren)

##########################################################################
###########  Visualization of the correlated profile    ##################
##########################################################################

fig, ax1 = plt.subplots()
ax1.tick_params(direction='in', top = True, right = True, length=7)
plt.rcParams['font.size'] = '17'
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
	label.set_fontsize(17)
img3 = ax1.contourf(Kz, Kp, np.abs(Psi2pen(Kz, Kp))**2,  20, cmap='seismic')
ax1.set_xlabel(r"$k$")
ax1.set_ylabel(r"$k '$")

cbar = plt.colorbar(img3,cmap='seismic')
cbar.set_label(r"$|\psi _{2p}(k, k')|^{2}$",size=17)
plt.show()

##########################################################################
########  Visualization of the correlated profile with k/k_{0}   #########
##########################################################################


k22 = (1/k0)*np.arange(k0 - deltakev*(Modes/2), k0 + deltakev*(Modes/2), deltakev)
k23 = (1/k0)*np.arange(k0 - deltakev*(Modes/2), k0 + deltakev*(Modes/2), deltakev)

#k22 = np.linspace(1 - 3.6*sigmaev, 1 + 3.6*sigmaev, Modes)
#k23 = np.linspace(1 - 3.6*sigmaev, 1 + 3.6*sigmaev, Modes)

K22, K23 = np.meshgrid(k22, k23)


def Psi2pennorm(x, x1):
    return np.exp(-((k0**2)*(x - 1)**2)/(4 * sigmaev**2)) * (np.pi * (ss*sigmaev)**2)**(-0.25) * np.exp(-(k0**2)*((x + x1 - 2)**2)/(4*((ss*sigmaev)**2))) * expivec(k0 * x) * expivec(k0 * x1)  

fig, ax1 = plt.subplots()
ax1.tick_params(direction='in', top = True, right = True, length=7)
plt.rcParams['font.size'] = '17'
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
	label.set_fontsize(17)
img4 = ax1.pcolormesh(K22, K23, np.abs(Psi2pennorm(K22, K23))**2, shading='gouraud', cmap = cm.seismic)
ax1.set_xlabel(r"$k/k_{0}$")
ax1.set_ylabel(r"$k '/k_{0}$")
ax1.set_title(r"$\quad \sigma _{s} = 0.05\sigma$")

cbar = plt.colorbar(img4,cmap='seismic')
cbar.set_label(r"$|\psi _{2p}(k, k')|^{2}$",size=17)
#plt.savefig("/home/christiandavid/Desktop/UCL_PhD/corr_photons_001sigma.pdf", format="pdf", dpi=300)
plt.show()

##########################################################################
################## Set of differential equations #########################
##########################################################################


psi2p = np.zeros((len(kz), len(kz)), dtype=complex)
psi1pm = np.zeros((len(kz), Nlevels), dtype=complex)
psie = np.zeros((Nlevels, 1), dtype=complex)
dpsi2p = np.zeros((len(kz), len(kz)), dtype=complex)
dpsi1pm = np.zeros((len(kz), Nlevels), dtype=complex)
dpsie = np.zeros((Nlevels, 1), dtype=complex)


#Initialization of the vector psi lists 
listapsi2p = []
listapsi1pm = []
listapsie = []

#Definition of the vector gamma
gammavec = np.sqrt((gammarev/(2*np.pi)) * lista2) 
gammavector = gammavec.reshape(Nlevels, 1)

#Definition of the matrix of sqrt(Fnn)
Fnngamm = np.sqrt((gammarev/np.pi) * Fnn)

#Initialization of \gamma ^{(gm)}
gammagm = np.zeros((len(kz), Nlevels), dtype=complex)

#Defifinition of \gamma ^{(gm)}
for i in range(len(kz)):
    gammagm[i] = np.sqrt(((2*gammarev)/np.pi) * lista2) 

#####################################################################################################
############ Definition of the function which contains the set of differential equations ############
#####################################################################################################

def odes(t, psiv):
    
    listapsi2p = []
    listapsi1pm = []
    listapsie = []
    
    #Definition of the part of psiv which corresponds to the matrix psi2p
    
    for i in np.arange(0, (len(kz)) * (len(kz))):
        listapsi2p.append(psiv[i])

    listapsi2p = np.array(listapsi2p)

    psi2p = listapsi2p.reshape((len(kz), len(kz))) #Definition of matrix psi2p in terms of vector components of psiv

    #####################################################################################################
    #Definition of the part of psiv which corresponds to the matrix psi1pm
    
    for i in np.arange((len(kz)) * (len(kz)), ((len(kz)) * (len(kz))) + ((len(kz)) * Nlevels)):
        listapsi1pm.append(psiv[i])

    listapsi1pm = np.array(listapsi1pm)

    psi1pm = listapsi1pm.reshape((len(kz), Nlevels)) #Definition of matrix psi1om in terms of vector components of psiv
    ####################################################################################################

    #Definition of the part of psiv which corresponds to the matrix psie
    
    for i in np.arange(((len(kz)) * (len(kz))) + ((len(kz)) * Nlevels), ((len(kz)) * (len(kz))) + ((len(kz)) * Nlevels) + Nlevels):
        listapsie.append(psiv[i])

    listapsie = np.array(listapsie)

    psie = listapsie.reshape((Nlevels, 1)) #Definition of vector psie in terms of vector components of x
    ####################################################################################################

    ##################################################################
    ################ Definition of matrix d/dt psi2p #################
    ##################################################################
    
    psi1pmgamma1  = np.matmul(psi1pm, gammavector)
    for alpha in range(len(kz)):
        for beta in range(len(kz)):
            dpsi2p[alpha, beta] = -1j*(1/sigmaev)*(kz[alpha] + kp[beta])*psi2p[alpha, beta] - 1j*(1/sigmaev)*psi1pmgamma1[alpha] - 1j*(1/sigmaev)*psi1pmgamma1[beta]
    ##################################################################

    ##################################################################
    ################ Definition of matrix d/dt psi1pm ################
    ##################################################################
    
    psip2gammagm = np.matmul(psi2p, gammagm)
    gammamepsie = np.matmul(Fnngamm, psie)
    for alpha in range(len(kz)):
        for n in range(Nlevels):
            dpsi1pm[alpha, n] = -1j*(1/sigmaev)*(kz[alpha] + omegalnu(1, n))*psi1pm[alpha, n] - 1j*(deltakev/sigmaev)*psip2gammagm[alpha, n] - 1j*(1/sigmaev)*gammamepsie[n]
    ##################################################################

    ##################################################################
    ################ Definition of vector d/dt psie ##################
    ##################################################################  
    
    psi1pmgammame = np.matmul(psi1pm, Fnngamm)
    sumcolumns = psi1pmgammame.sum(axis=0)
    for n in range(Nlevels):
        dpsie[n] = -1j*(1/sigmaev)*omegalnu(2, n)*psie[n] - 1j*(deltakev/sigmaev)*sumcolumns[n]
    ##################################################################

    #Definition of the vector dy which contains the derivatives
    dy = np.array(np.append(np.append(dpsi2p.flatten(), dpsi1pm.flatten()), dpsie.flatten()))    

    return dy

##########################################################################
####################  Initial conditions  ################################
##########################################################################

###### Initial conditions of psi2p ######
psi2p0 = np.zeros((len(kz), len(kz)), dtype=complex)

psi2p0 = Psi2pen(Kz, Kp)

###### Initial conditions of psi1pm ######
psi1pm0 = np.zeros((len(kz), Nlevels), dtype=complex)

###### Initial conditions of psie #######
psie0 = np.zeros((Nlevels, 1), dtype=complex)

##### Total initial conditions
psiv0 = np.array(np.append(np.append(psi2p0.flatten(), psi1pm0.flatten()), psie0.flatten()))

print("==========================================================================")
print("====================  Initial conditions  ================================")
print("==========================================================================")

print("Initial conditions: ")
print(psiv0)

##########################################################################

##########################################################################
############################  Solver  ####################################
##########################################################################

print("==========================================================================")
print("=======================  Solver started...  ==============================")
print("==========================================================================")

r = ode(odes)
r.set_integrator('zvode', method='adams')
r.set_initial_value(psiv0, t0)

lista1 = []

while r.successful() and r.t < t1:
    print(r.t)
    lista1.append(r.integrate(r.t+dt))

lista1 = np.array(lista1)

tii = np.linspace(t0, t1, np.shape(lista1)[0])


np.savetxt('/home/christiandavid/Desktop/UCL_PhD/solutions_MO_entangled_ss001_64modes_corrected.txt', lista1)
print(np.shape(lista1))


########################################
fig, ax1 = plt.subplots()
ax1.tick_params(direction='in', top = True, right = True, length=7)
plt.rcParams['font.size'] = '17'
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
	label.set_fontsize(17)

plt.plot(tii, np.abs(lista1[:, -Nlevels + 0])**2, color = "rosybrown", label = r"$\nu = 0$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 1])**2, color = "red", label = r"$\nu = 1$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 2])**2, color = "salmon", label = r"$\nu = 2$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 3])**2, color = "sienna", label = r"$\nu = 3$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 4])**2, color = "saddlebrown", label = r"$\nu = 4$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 5])**2, color = "darkorange", label = r"$\nu = 5$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 6])**2, color = "goldenrod", label = r"$\nu = 6$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 7])**2, color = "gold", label = r"$\nu = 7$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 8])**2, color = "olive", label = r"$\nu = 8$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 9])**2, color = "darkolivegreen", label = r"$\nu = 9$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 10])**2, color = "lawngreen", label = r"$\nu = 10$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 11])**2, color = "green", label = r"$\nu = 11$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 12])**2, color = "mediumspringgreen", label = r"$\nu = 12$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 13])**2, color = "mediumaquamarine", label = r"$\nu = 13$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 14])**2, color = "aquamarine", label = r"$\nu = 14$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 15])**2, color = "teal", label = r"$\nu = 15$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 16])**2, color = "cyan", label = r"$\nu = 16$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 17])**2, color = "deepskyblue", label = r"$\nu = 17$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 18])**2, color = "steelblue", label = r"$\nu = 18$", linestyle = "dashdot")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 19])**2, color = "dodgerblue", label = r"$\nu = 19$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 20])**2, color = "blue", label = r"$\nu = 20$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 21])**2, color = "blueviolet", label = r"$\nu = 21$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 22])**2, color = "magenta", label = r"$\nu = 22$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 23])**2, color = "hotpink", label = r"$\nu = 23$")
plt.plot(tii, np.abs(lista1[:, -Nlevels + 24])**2, color = "purple", label = r"$\nu = 24$")

plt.xlabel(r"$r \sigma $", fontsize=17)
plt.ylabel(r"$\langle e _{\nu} \rangle$", fontsize=17)
#ax1.set_title(r"$\quad \sigma _{s} =1.5 \sigma$ for MO")
plt.legend(bbox_to_anchor=(1.001, 1.01), loc="upper left", fontsize=7, fancybox=True, framealpha=1, borderpad=1, edgecolor="black")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.909, top=0.9)
#ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
plt.show()


