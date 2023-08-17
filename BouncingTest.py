""" Bouncing sphere """

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from scipy import special
import time
from tqdm import tqdm
import os
import sys

##########  NUMERICAL PARAMETERS ##########
# Saving Parameters
N_save = 100 # save full profile every N_save time step
Stokes_array = np.array([10,13,16,20,25,30,40,50,63,80,100,126,160,200,250,300,400,500,630,800,1000]) # Stokes numbers investigated
NR_array = np.array([600, 600, 600, 600, 600, 600, 700, 800, 900, 1000, 1100, 1200, 1400, 1500, 1600, 1700, 1900, 2000, 2200, 2400, 2600]) # Number of radial grid points in each simulation

# External Parameters - here we set to the first Stokes number of the list (10)
Stokes = Stokes_array[0]
NR = NR_array[0]

# Saving
#os.mkdir('Stokes'+'%.0f' % Stokes)
#os.mkdir('Stokes'+'%.0f' % Stokes+'/Fields')


##########  AXIS DEFINITION ##########
# Space
LR = 6.0 # Domain size in r  [0, L]
dr = LR/NR # grid size

# Time axis.
dt = 0.001 # time increment - In the data use in the paper, we took dt = 1e-4 instead. 
NT = 10000 # number of time steps.

# Initialization
NT_ini = 100 # number of time steps in the initialization phase
t1 = -NT_ini*dt # t_ini in the Appendix.

# Axis
r = np.linspace(0, LR , NR) # We take a uniform distribution of point in r
Time_ini = np.linspace(t1, 0, NT_ini+1) # Initialization time axis.
Time = np.linspace(0, dt*NT, NT+1) # Time axis


########## Initial position ###########
D0 = 1
D_ini = D0 - (Time_ini-Time_ini**2/(2*t1)) # sphere position during initialization
V_ini = -1+Time_ini/t1 # sphere velocity during initialization



######### Main Function ###########
def LinearOperator(y_old, D_old, V_old, KernelMatrix):
    ''' Linear Matrix L of the linear system L \cdot y = rhs.  '''
    L = np.zeros((2*NR,2*NR))
    w_old, p_old = y_old[:NR], y_old[NR:2*NR]



    ############## Elasticity  ##############
    np.fill_diagonal(L[:NR, :NR], np.ones(NR)) # L_{i,i}
    L[:NR, NR:2*NR] = KernelMatrix



    ############## Boundary conditions ##############
    L[NR,NR], L[NR,NR+1] = 1, -1
    L[2*NR-1,2*NR-1] = 1

    ############## Thin-film equation ##############
    h = D_old + r**2/2 - w_old
    alpha = 3 * (r[1:-1] - (w_old[2:]-w_old[1:-1])/dr ) + h[1:-1]/r[1:-1]
    diagonal_pressure, overdiagonal_pressure, underdiagonal_pressure = np.ones(NR-2), np.ones(NR-2), np.ones(NR-2)
    diagonal_pressure[:] = (2*h[1:-1]**3 + h[1:-1]**2 * alpha[:]*dr)
    overdiagonal_pressure[:] = -(h[1:-1]**3 + h[1:-1]**2 * alpha[:]*dr)
    underdiagonal_pressure[:] = -h[1:-1]**3

    np.fill_diagonal(L[NR+1:2*NR-1, NR+1:2*NR-1], diagonal_pressure) # L_{i,i} = partial L_i / partial p_i
    np.fill_diagonal(L[NR+1:2*NR-1, NR+2:2*NR-0], overdiagonal_pressure) # L_{i,i+1} = partial L_i / partial p_{i+1}
    np.fill_diagonal(L[NR+1:2*NR-1, NR+0:2*NR-2], underdiagonal_pressure) # L_{i,i-1} = partial L_i / partial p_{i-1}

    # Deformation terms
    diagonal_deformation = -dr**2 / (Stokes*dt)
    np.fill_diagonal(L[NR+1:2*NR-1, 1:NR-1], diagonal_deformation) # partial L_i / partial w_i

    return L





def rhs(y_old, V_old):
    ''' right hand side of the linear system L \cdot y = rhs.  '''
    RHS = np.zeros(2*NR)
    w_old, p_old = y_old[:NR], y_old[NR:2*NR]
    RHS[NR+1:2*NR-2] = -(w_old[1:NR-2] + V_old*dt) * dr**2 / (Stokes*dt)
    return RHS



def UpdateDistance_Velocity(y_new, D_old, V_old):
    ''' Return the new velocity and distance '''
    w_new, p_new = y_new[:NR], y_new[NR:2*NR]
    V_new = V_old + 4 * dt * sum(p_new[:]*r[:]*dr)
    D_new = D_old + V_new*dt
    return D_new, V_new





print('Initialization : ')
KernelMatrix = np.zeros((NR, NR))
print('Kernel Computation')
for i in tqdm(range(0, NR)):
    KernelMatrix[i, 0] = scipy.integrate.quadrature(lambda u: u/(r[i]+u)*scipy.special.ellipk(4*r[i]*u/(r[i]+u)**2)/(np.pi**2/8) , 0, dr/2)[0]
    for j in range(1,NR):
        KernelMatrix[i, j] = scipy.integrate.quadrature(lambda u: u/(r[i]+u)*scipy.special.ellipk(4*r[i]*u/(r[i]+u)**2)/(np.pi**2/8) , r[j]-dr/2, r[j]+dr/2)[0]

# Initialization
y_ini, Deformation_ini, Pressure_ini = np.zeros((2*NR, NT_ini)), np.zeros((NR, NT_ini)), np.zeros((NR, NT_ini))
for i in tqdm(range(0, NT_ini-1)):
    y_ini[:,i+1] = np.linalg.solve(LinearOperator(y_ini[:,i], D_ini[i], V_ini[i], KernelMatrix), rhs(y_ini[:,i], V_ini[i]))
    Deformation_ini[:,i+1], Pressure_ini[:,i+1] = y_ini[:NR,i+1], y_ini[NR:2*NR,i+1]
print('Initialization == DONE')



y, Deformation, Pressure = np.zeros((2*NR, NT+1)), np.zeros((NR, NT+1)), np.zeros((NR, NT+1))
Deformation[:,0], Pressure[:,0] = Deformation_ini[:,-1], Pressure_ini[:,-1]
y[:NR, 0], y[NR:, 0] = Deformation[:,0], Pressure[:,0]
D, V, Force, EnergyElastic, Central_Film_thikness, MinFilmThickness = np.zeros(NT+1), np.zeros(NT+1), np.zeros(NT+1), np.zeros(NT+1), np.zeros(NT+1), np.zeros(NT+1)
D[0], V[0] = D_ini[-1], V_ini[-1]
print('Main loop : ')
for i in tqdm(range(0, NT)):
    y[:,i+1] = np.linalg.solve(LinearOperator(y[:,i], D[i], V[i], KernelMatrix), rhs(y[:,i], V[i]))
    Deformation[:,i+1], Pressure[:,i+1] = y[:NR,i+1], y[NR:2*NR,i+1]
    D[i+1], V[i+1] = UpdateDistance_Velocity(y[:,i+1], D[i], V[i])
    EnergyElastic[i+1] = -2*np.sum(Pressure[:,i+1]*Deformation[:,i+1]*r[:]*dr)
    Force[i+1] = 4*np.sum(Pressure[:,i+1]*r[:]*dr)
    Central_Film_thikness[i+1] = D[i+1] - Deformation[0,i+1]
    MinFilmThickness[i+1] = min(D[i+1] + r[:]**2/2 - Deformation[:,i+1])

    #Saving
    #np.savetxt('Stokes'+'%.0f' % Stokes+'/Global_St'+'%.0f' % Stokes+'.txt', np.c_[Time[:i+1], D[:i+1], V[:i+1], Force[:i+1], EnergyElastic[:i+1], Central_Film_thikness[:i+1], MinFilmThickness[:i+1]], delimiter=' ',fmt=['%-.9g', '%-.9g', '%-.9g', '%-.9g', '%-.9g', '%-.9g', '%-.9g'])
    #if i%N_save == 0:
    #    np.savetxt('Stokes'+'%.0f' % Stokes+'/Fields/t'+'%.3f' % Time[i]+'.txt', np.c_[r, Deformation[:,i], Pressure[:,i]], delimiter=' ',fmt=['%-.9g', '%-.9g', '%-.9g'])
    if D[i+1] > 1.21:
        break

print('Bouncing == DONE')
