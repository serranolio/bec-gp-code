# -*- coding: utf-8 -*-
"""
I. PARAMETERS DEFINITION
GP simulation
geometry: 1D
system: 3-component BEC
author: Federico Serrano
Physics and Astronomy Department
Washington State University
""" 

import numpy as np
import os
from numpy import pi


''' 
1.Constants -----------------------------------------------------------
'''                                                             
m_ua = 86.909187
m_si = m_ua/6.022140857e23/1000
hbar_si = 1.054571800139113e-34
a_si = 5.2917721067e-11

f_recoil = 1960.0
n_atoms = 1.93369e5

fz = 27.99
fx = 176.22
fy = 198.53

wz = fz / f_recoil
wx = np.sqrt(fx*fy) / f_recoil


e_unit = 2*pi*hbar_si * f_recoil
l_unit = hbar_si/np.sqrt(2*m_si*e_unit)
t_unit = hbar_si/e_unit

'''
2. Parameters ---------------------------------------------------------
'''
# hamiltonian parameters
k_l = 0.74

params = {
          'k_l': k_l,
          }

# calculated parameters
a = 100.4*a_si/l_unit
g = 8*pi*a
w3 = (fx*fy*fz)**(1/3)/f_recoil
mu = (15*w3**3*n_atoms*g/(64*pi))**(2/5)
gn = (64/105)*4*pi/(w3**3)*mu**(7/2)/g/n_atoms
#g_1d = 8/3*gn**(3/2)/wz
g_1d = 8/3*mu**(3/2)/wz

rx = np.sqrt(4*mu)/wx
rz = np.sqrt(4*mu)/wz

'''
3. System's geometry --------------------------------------------------
'''
nz = 2**9
lz = 2.5*rz

dz = lz/nz
z = np.arange(nz)*dz - (lz-dz)/2

kz = 2*pi*np.fft.fftfreq(nz, d=dz)

print(f'Initialize system with: \n' +
      f'chemical potential --> {mu} \n' +
      f'size in z --> {lz} \n' +
      f'size in kz --> {kz.max()} \n' +
      f'grid points nz--> {nz} \n' +
      f'Thomas-Fermi radii Rx, Rz--> {rx}, {rz} \n')

'''
4. External potentials ------------------------------------------------
'''
U = (wz*z)**2/4

'''
5. Output format ------------------------------------------------------
'''

system_params_str = ''
for keys, values in params.items():
    system_params_str += (keys + '_%.2f_' % values)

params_str = (f'nz_{nz:.0f}_lz_{lz:.0f}' +
              f'_n_atoms_{n_atoms:.0f}_' + system_params_str +
              f'.npy')

'''
7. Assertions ---------------------------------------------------------
'''

assert dz < 1/np.sqrt(gn)
