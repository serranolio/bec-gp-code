# -*- coding: utf-8 -*-
"""
I. PARAMETERS DEFINITION
GP simulation
geometry: 3D
system: 2-component BEC
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

wx = fx / f_recoil
wy = fy / f_recoil
wz = fz / f_recoil


e_unit = 2*pi*hbar_si * f_recoil
l_unit = hbar_si/np.sqrt(2*m_si*e_unit)
t_unit = hbar_si/e_unit

'''
2. Parameters ---------------------------------------------------------
'''
# Hamiltonian parameters
omega_r = 2.7
q_zeeman = 7.189099499024451e3/f_recoil
k_l = np.sqrt(1 - (omega_r/4)**2)
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

rx = np.sqrt(4*mu) / wx
ry = np.sqrt(4*mu) / wy
rz = np.sqrt(4*mu) / wz

'''
3. System's geometry --------------------------------------------------
'''

nx = 2**5
ny = 2**5
nz = 2**8

lx = 2.5 * rx
ly = 2.5 * ry
lz = 2.5 * rz

dx = lx / nx
dy = ly / ny
dz = lz / nz

x = np.arange(nx) * dx - (lx - dx) / 2
y = np.arange(ny) * dy - (ly - dy) / 2
z = np.arange(nz) * dz - (lz - dz) / 2

kx = 2*pi*np.fft.fftfreq(nx, d=dx)
ky = 2*pi*np.fft.fftfreq(ny, d=dy)
kz = 2*pi*np.fft.fftfreq(nz, d=dz)

x_, y_, z_ = np.meshgrid(x, y, z)
kx_, ky_, kz_ = np.meshgrid(np.fft.fftshift(kx),
                            np.fft.fftshift(ky),
                            np.fft.fftshift(kz))

print(f'Initialize system with: \n' +
      f'chemical potential --> {mu} \n' +
      f'size in z --> {lz} \n' +
      f'size in kz --> {kz.max()} \n' +
      f'grid points nz--> {nz} \n' +
      f'Thomas-Fermi radii Rx, Rz--> {rx}, {rz} \n')

'''
4. External potentials ------------------------------------------------
'''
U = ((wx*x_)**2/4 + (wy*y_)**2/4 + (wz*z_)**2/4).reshape(nx*ny*nz)

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
assert dx < 1 / np.sqrt(gn)
assert dy < 1 / np.sqrt(gn)
assert dz < 1 / np.sqrt(gn)

