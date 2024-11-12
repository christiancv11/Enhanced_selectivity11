import sys
sys.path.append('/home/ucapcdr/Scratch/ETPA')
import time
import numpy as np
import mpmath
from mpmath import *
import scipy
from scipy import integrate
import cmath
#from numba import jit
from scipy.integrate import ode
mp.dps = 45; mp.pretty = True

np.set_printoptions(threshold=sys.maxsize)
# get the start time
st = time.time()

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
Modes = 4001 #Number of modes for the incoming field
lex = 18 #Level of resonance

print("Number of levels for each potential: ", Nlevels)
print("Number of modes for the field: ", Modes)
print("Level of resonance: ", lex)
print("==========================================================================")

##########################################################################

##########################################################################
################## Constants for the photons profile #####################
##########################################################################


#Value of the spectral width of the wave packet \sigma = \sigma _{r}^{-1} in Hz
sigma = 10e12

#Value of the spectral width of the wave packet \sigma = \sigma _{r}^{-1} in eV -> 1Hz = 4.135 58 x 10^-15 eV

sigmaev = sigma * 4.13558e-15

#Constant for entanglement

ss = 0.1
print("Constants for the photons profile : ")
print("sigma: ", sigma)
print("sigma in eV: ", sigmaev)
print("ss: ", ss)
print("==========================================================================")
##########################################################################

##########################################################################
################### Constants for the simulations ########################
##########################################################################

t0 = -50.0
t1 = 50.0
dt = 0.01

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

#Definition of the relaxation rate \gamma = \gamma _{m_{\nu}} = \gamma _{e _{\nu \nu'}} in Hz
gammar = 6e6
print("relaxation rate gamma: ", gammar)

#Definition of the relaxation rate \gamma = \gamma _{m_{\nu}} = \gamma _{e _{\nu \nu'}} in eV
gammarev = gammar * 4.13558e-15
print("relaxation rate gamma in eV: ", gammarev)
print("==========================================================================")
##########################################################################

###############################################################
###### Definition of the vibrational eigen-energies ###########
###############################################################

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
    return np.sqrt((2*potdepth(l))/((al(l)**2) * mu))

#Definition of \chi _{\ell}

def chil(l):
    return 1/np.sqrt(8 * (al(l)**2) * potdepth(l) * mu)
    
    
# Definition of \omega _{\ell _{\nu}}
def omegalnu_func(l, nu):
    return epsilonl(l) + (omegal(l) * (nu + 0.5)) - (omegal(l) * chil(l) * (nu + 0.5) ** 2)
    
omegalnu = np.zeros([3, Nlevels])

for l in range(3):
    for nu in range(Nlevels):
        omegalnu[l, nu] = omegalnu_func(l, nu) 
    
    
#########################################################################
## Definition of the two-photon joint amplitude for correlated pairs  ###
#########################################################################

wlength = float(4*np.pi/omegalnu[2, lex]) #Wavelength in ev^-1
k0 = float(2*np.pi / wlength) #Wave number in ev -> k_{0}
sigmaev = float(sigmaev)
r0 = float(t0/sigmaev) #spatial center position of the wave packet at t0

print("==========================================================================")
print("=================   Parameters of the correlated pairs  ==================")
print("==========================================================================")

#Energy of incident field
print("Energy of incident field in eV: ")
print("k0: ", k0, "2k0: ", 2*k0, "r0: ", r0)

#Value of \delta k in Hz
deltak = 100 * (10**9)
print("deltak: ", deltak)

#Value of \delta k in eV
#deltakev = deltak * 4.13558e-15
deltakev = (9 * sigmaev)/Modes
print("delta k in eV: ", deltakev)

kz = np.linspace(k0 - 4.5*sigmaev, k0 + 4.5*sigmaev, Modes) #Array of all k

kp = np.linspace(k0 - 4.5*sigmaev, k0 + 4.5*sigmaev, Modes) #Array of all k'

Kz, Kp = np.meshgrid(kz, kp) #Multilinear array of all possible combinations of k and k'

noren = np.sqrt(2 * np.pi * sigmaev * ss * sigmaev) #Normalization constant
#noren = np.sqrt(2 * np.pi * sigmaev**2)

#Definition of the two-photon joint amplitude of an entangled photon pair with energy anticorrelation
def Psi2pen(k, k1):
    return np.exp(-((k - k0)**2)/(4 * sigmaev**2)) * np.exp(-((k + k1 - 2*k0)**2)/(4 * ss**2 * sigmaev**2)) * (np.cos((k + k1)*r0) - 1j*np.sin((k + k1)*r0)) * (1/noren)     
    
# ----------------------------------------------------------------------------------------------------#    
    
# Non-symmetric correlated profile:    
#np.exp(-((k - k0)**2)/(4 * sigmaev**2)) * np.exp(-((k + k1 - 2*k0)**2)/(4 * ss**2 * sigmaev**2)) * (np.cos((k + k1)*r0) - 1j*np.sin((k + k1)*r0)) * (1/noren)     
    
# Symmetric correlated profile:    
#np.exp(-(((k - k1)*0.5)**2)/(4 * sigmaev**2)) * np.exp(-((k + k1 - 2*k0)**2)/(4 * ss**2 * sigmaev**2)) * (np.cos((k + k1)*r0) - 1j*np.sin((k + k1)*r0)) * (1/noren)       
    
# Uncorrelated profile:    
#np.exp(-((k - k0)**2)/(4 * sigmaev**2)) * np.exp(-((k1 - k0)**2)/(4 * sigmaev**2)) * (np.cos((k + k1)*r0) - 1j*np.sin((k + k1)*r0)) * (1/noren)
  
# ----------------------------------------------------------------------------------------------------#    

##########################################################################
########:::::::::::::::  Franck-Condon Factors  :::::::::::::::::#########
##########################################################################

#Importing of files

#Franck-Condon Factors g->m
lista2 = []

nus = np.array(range(Nlevels))

lista2 = np.loadtxt('/lustre/scratch/scratch/ucapcdr/ETPA/Ff_Factors_gm_63.txt')
lista2 = np.array(lista2, dtype=float)

lista22 = []

for i in range(Nlevels):
    lista22.append(lista2[i])
    
lista2 = lista22

lista2 = np.array(lista2, dtype=float)

#Franck-Condon Factors m->e
Fnn = np.zeros((Nlevels, Nlevels))

Fnn = np.loadtxt('/lustre/scratch/scratch/ucapcdr/ETPA/Ff_Factors_me_63.txt')

Fnn1 = np.zeros((Nlevels, Nlevels))

for i in range(Nlevels):
    for j in range(Nlevels):
        Fnn1[i, j] = Fnn[i, j]
        
Fnn = Fnn1

##########################################################################
################## Set of differential equations #########################
##########################################################################

#Definition of the matrix ka + kb
sumk = np.zeros((Modes, Modes), dtype=complex)


for i in range(Modes):
    for j in range(Modes):
        sumk[i, j] = kz[i] + kp[j]

sumk = sumk.flatten()

#Definition of the matrix ka + wmnu

sumkwm = np.zeros((Modes, Nlevels), dtype=complex)

for i in range(Modes):
    for n in range(Nlevels):
        sumkwm[i, n] = kz[i] + omegalnu[1, n]

sumkwm = sumkwm.flatten()


#Definition of the vector gamma
gammanu = np.sqrt(gammarev * deltakev * (lista2/np.pi)) 

#Definition of the matrix of sqrt(Fnn)
gammanunup = np.sqrt((gammarev/np.pi) * deltakev * Fnn)

def odes(t, psi):
    
    # Definition of the vector psi2p in terms of the components of psi
    psi2p = psi[0 : Modes ** 2]
        
    # Definition of the vector psi1p in terms of the components of psi
    psi1pm = psi[Modes ** 2 : Modes * (Modes + Nlevels)]
       
    # Definition of the vector psi0e in terms of the components of psi
    psi0e = psi[Modes * (Modes + Nlevels) : Modes * (Modes + Nlevels + 1)]

    ##################################################################
    ###############  Definition of array d/dt psi2p  #################
    ##################################################################
   
    psi1pm_mat = psi1pm.reshape((Modes, Nlevels))
   
    def f1(k):
        return np.dot(gammanu, psi1pm_mat[k, :])
    
    f1_list = np.fromiter((f1(k) for k in range(Modes)), dtype=np.complex128, count=Modes)
    
    sumf1 = f1_list[:, np.newaxis] + f1_list
           
    dpsi2p = -1j * (1 / sigmaev) * sumk * psi2p - (1j / np.sqrt(2)) * (1 / sigmaev) * sumf1.flatten()

    ##################################################################
    ###############  Definition of array d/dt psi1p  #################
    ##################################################################
    
    psi2p_mat = psi2p.reshape((Modes, Modes))
    
    f21_list = np.sum(psi2p_mat[:Modes, :], axis=1)
    
    f3_list = np.dot(gammanunup[:Nlevels, :], psi0e)
   
    gammanu_broadcasted = gammanu[:Nlevels, np.newaxis].T
    sqrt_2_inv = 1 / np.sqrt(2)
    f2 = gammanu_broadcasted * f21_list[:, np.newaxis] + sqrt_2_inv * np.tile(f3_list, (Modes, 1))

    dpsi1pm = -1j * (1 / sigmaev) * sumkwm * psi1pm - 1j * (1 / sigmaev) * np.sqrt(2) * f2.flatten()    
       
    ##################################################################
    ###############  Definition of array d/dt psi0e  #################
    ##################################################################
    
    psi1pm_mat = psi1pm.reshape((Modes, Nlevels))

    f5_list = np.sum(psi1pm_mat, axis=0) 

    dpsi0e = -1j * (1 / sigmaev) * omegalnu[2, :] * psi0e - 1j * (1 / sigmaev) * np.dot(gammanunup, f5_list)
    
    #Definition of the vector dy which contains the derivatives    
    dy = np.concatenate((dpsi2p, dpsi1pm, dpsi0e)) 

    return dy

##########################################################################
####################  Initial conditions  ################################
##########################################################################

###### Initial conditions of psi2p ######
psi2p0 = np.zeros((Modes, Modes), dtype=complex)

for i in range(Modes):
    for j in range(Modes):
        psi2p0[i, j] = Psi2pen(kz[i], kp[j])

###### Initial conditions of psi1pm ######
psi1pm0 = np.zeros((Modes, Nlevels), dtype=complex)

###### Initial conditions of psie #######
psie0 = np.zeros((Nlevels, 1), dtype=complex)

##### Total initial conditions
psiv0 = np.array(np.append(np.append(psi2p0.flatten(), psi1pm0.flatten()), psie0.flatten()))

#print("==========================================================================")
#print("====================  Initial conditions  ================================")
#print("==========================================================================")

norinitstate = np.sum(np.abs(psiv0)**2)

psiv0nor = psiv0/np.sqrt(norinitstate)

##########################################################################

##########################################################################
############################  Solver  ####################################
##########################################################################

print("==========================================================================")
print("=======================  Solver started...  ==============================")
print("==========================================================================")


methodsolver = 'adams'

r = ode(odes)
r.set_integrator('zvode', method=methodsolver)
r.set_initial_value(psiv0nor, t0)

lista1 = []

while r.successful() and r.t < t1:
    print(r.t)
    lista1.append(r.integrate(r.t+dt)[Modes * (Modes + Nlevels) : Modes * (Modes + Nlevels + 1)])

##########################################################################
##########################  Creation of the files ########################
##########################################################################

#Array which contains all the solutions
lista1 = np.array(lista1)

#Temporal array
tii = np.linspace(t0, t1, np.shape(lista1)[0])

#Array which contains the final row of the solutions - to be the initial conditions in the next running
flist = np.array(lista1[-1, :])

#Array which contains the solutions of the population of the 25 first excited levels
#norfinalstate = np.sum(np.abs(lista1[-1, :])**2)
norfinalstate = 1

list25excited = np.zeros((25, np.shape(lista1)[0]), dtype=complex)

for i in range(25):
    list25excited[i] = (1/norfinalstate) * np.abs(lista1[:, -Nlevels + i])**2

list25excited = np.array(list25excited)

#Definition of the name of the file which contains all the solutions
name_solutions = f'/lustre/scratch/scratch/ucapcdr/ETPA/solutions_MO/solutions_MO_entangled_ss{ss}_{Modes}modes_{methodsolver}_{t0}to{t1}_dt{dt}_r0_nonsym_fsolverp.txt'

#Definition of the name of the file which contains the final row of the solutions
name_initial_conditions = f'/lustre/scratch/scratch/ucapcdr/ETPA/initialconditions_MO/initial_conditions_MO_entangled_ss{ss}_{Modes}modes_{methodsolver}_{t0}to{t1}_dt{dt}_r0_nonsym_fsolverp.txt'

#Definition of the name of the file which contains the solution of the 25 first excited levels
name_populations = f'/lustre/scratch/scratch/ucapcdr/ETPA/populations_MO/populations25_MO_entangled_ss{ss}_{Modes}modes_{methodsolver}_{t0}to{t1}_dt{dt}_r0_nonsym_fsolverp.txt'



#Save the total array of all solutions
np.savetxt(name_solutions, lista1)

#Save the final row of the solutions to be the initial conditions in the next running
np.savetxt(name_initial_conditions, flist)

#Save the 25 first excited levels
np.savetxt(name_populations, list25excited)

#print(np.shape(lista1))

#print('Final shape: ', np.shape(lista1))
#print("==========================================================================")
#print("Final vector: ")
#print(np.abs(lista1[:, -Nlevels + 18])**2)
#print("==========================================================================")
#print("Final data:")
#print("Number of levels for each potential: ", Nlevels)
#print("Number of modes for the field: ", Modes)
#print("Level of resonance: ", lex)
#print("ss: ", ss)
#print("Energy of incident field in eV: ")
#print("k0: ", k0, "2k0: ", 2*k0, "r0: ", r0)
#print("Simulation parameters: ")
#print("t0: ", t0)
#print("t1: ", t1)
#print("dt: ", dt)
#print("Final value of level 18: ", (1/norfinalstate) * (np.abs(lista1[:, -Nlevels + 18])**2)[-1])
#print("==========================================================================")



# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st

print('Execution time: ', elapsed_time, 'seconds')
print('Execution time: ', elapsed_time/60, 'minutes')
print('Execution time: ', elapsed_time/3600, 'hours')
print('Execution time: ', elapsed_time/86400, 'days')

print('norfinalstate: ', norfinalstate)

########################################

