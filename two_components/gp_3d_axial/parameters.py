# -*- coding: utf-8 -*-
"""
I. PARAMETERS DEFINITION
GP simulation
geometry: 3D-axial
system: 2-component BEC
author: Federico Serrano
Physics and Astronomy Department
Washington State University
""" 

import numpy as np
import discrete_hankel_transform as dht
import os
from numpy import pi
from scipy.special import jv, jn_zeros


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
# Hamiltonian parameters
omega_r = 2.7
k_l = np.sqrt(1 - (omega_r/4)**2)
q_zeeman = 7.189099499024451e3 / f_recoil
#k_l = 0.7378177281686853

params = {
          'omega_r': omega_r,
          'q_zeeman': q_zeeman,
          'k_l': k_l,
          }

# calculated parameters
a = 100.4*a_si/l_unit
g = 8*pi*a

w3 = (fx*fy*fz)**(1/3)/f_recoil
mu = (15*w3**3*n_atoms*g/(64*pi))**(2/5)
gn = (64/105)*4*pi/(w3**3)*mu**(7/2)/g/n_atoms

rx = np.sqrt(4*mu)/wx
rz = np.sqrt(4*mu)/wz

'''
3. system's geometry --------------------------------------------------
'''
nx, nz = 2**6, 2**9
lx, lz = 1.5*rx, 2.5*rz

dz = lz/nz
dx = 2*(lx/(jn_zeros(0, nx+1)[-1])/jv(1, jn_zeros(0, nx)))**2

dv = np.kron(2*pi*dz*dx, np.ones(nz))
dvk = (jn_zeros(0, nx+1)[-1]/lx**2)**2*(2*pi*nz/lz**2)*dv

x = jn_zeros(0, nx)*lx/(jn_zeros(0, nx+1)[-1])
z = np.arange(nz)*dz - (lz-dz)/2

kx = jn_zeros(0, nx)/lx
kz = 2*pi*np.fft.fftfreq(nz, d=dz)

x_, z_ = np.meshgrid(x, z, indexing='ij') 
kx_, kz_ = np.meshgrid(kx, np.fft.fftshift(kz), indexing='ij')

print(f'Initialize system with: \n' +
      f'chemical potential (Thomas-Fermi) --> {mu} \n' +
      f'size in x --> {lx} \n' +
      f'size in z --> {lz} \n' +
      f'size in kx --> {kx.max()} \n' +
      f'size in kz --> {kz.max()} \n' +
      f'grid points nx, nz--> {nx}, {nz} \n' +
      f'Thomas-Fermi radii Rx, Rz--> {rx}, {rz} \n' +
      f'grid spacing dx, dz --> {dx.min()}, {dz} \n' +
      f'healing length -- > {1/np.sqrt(gn)}')

'''
4. external potentials ------------------------------------------------
'''

U = ((wx*x_)**2/4 +                                                             
     (wz*z_)**2/4).reshape(nx*nz)

'''
5. transformation matrices --------------------------------------------
'''
print('creating (x, z) <-> (kx, kz) transformation matrix')
# Transverse kinetic energy
Tx = dht.idht(np.diag(kx**2) @ dht.dht(np.eye(nx), axis=0), axis=0)

# Axial kinetic energy
#Tz = np.fft.ifft(np.fft.fft(np.eye(nz)) @ np.diag(kz**2))
#Kz = np.fft.ifft(np.fft.fft(np.eye(nz)) @ np.diag(-kz))
print('creating (x, z) <-> (kx, kz) transformation matrix ---> done')

'''
6. output format ------------------------------------------------------
'''

system_params_str = ''
for keys, values in params.items():
    system_params_str += (keys + '_%.2f_' % values)

params_str = (f'nx_{nx:.0f}_nz_{nz:.0f}_lx_{lx:.0f}_lz_{lz:.0f}' +
              f'_n_atoms_{n_atoms:.0f}_' + system_params_str +
              f'.npy')

'''
7. assertions ---------------------------------------------------------
'''

assert dx.min() < 1/np.sqrt(gn)
assert dz < 1/np.sqrt(gn)
