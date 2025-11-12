# -*- coding: utf-8 -*-                                                         
"""                                                                             
III. EXECUTE SIMULATION
GP Simulation
geometry: 3D-axial
system: 2-component BEC
author: Federico Serrano
Physics and Astronomy Department
Washington State University
"""

import numpy as np
import pandas as pd
from system_functions import *

# path where 'execute_simulation.py' is stored
#directory_path = os.path.dirname(os.path.realpath(__file__))

# path where 'execute_simulation.py' is executed from
directory_path = os.getcwd()

os.makedirs("output", exist_ok=True)
os.makedirs("input", exist_ok=True)

assert os.path.isdir(directory_path + '/output')
assert os.path.isdir(directory_path + '/input') 

'''
1. simulation parameters ----------------------------------------------
'''

# ramp parameters
ramp_time_ms = 0
ramp_time = ramp_time_ms / 1000 / t_unit

simulation_time = 20e-3 / t_unit
#simulation_time = ramp_time
#simulation_time = (ramp_time + 42*np.sqrt(ramp_time))

delta_i =  5000 / f_recoil
delta_f = -100 / f_recoil

omega_l_i = 0.1
omega_l_f = 0.1

sample = 0

def detuning_ramp(t):
    delta = ((delta_i + (delta_f - delta_i)*t/ramp_time)*(t<=ramp_time)
             + delta_f*(t>ramp_time))
    return delta

def lattice_ramp(t):
    omega = ((omega_l_i + (omega_l_f - omega_l_i)*t/ramp_time)*(t<=ramp_time)
             + omega_l_f*(t>ramp_time))
    return omega

sample_str = (f'sample_{sample:.0f}_ramp_time_{ramp_time_ms:.2f}ms_'
              + f'delta_i_{delta_i*f_recoil:.0f}Hz_'
              + f'delta_f_{delta_f*f_recoil}Hz_'
              + f'omega_l_i_{omega_l_i:.2f}_omega_l_f_{omega_l_f:.2f}_')

out_file_name = (directory_path + 
                 '/output/wavefunction_' + sample_str + params_str)

'''
2. get ground state ---------------------------------------------------
'''

print('loading ground state')
try:
    psi_gs = np.load(directory_path
                     + f'/input/ground_state_omega_l_{omega_l_i:.2f}_'
                     + f'delta_{delta_i:.2f}_' 
                     + params_str)
    print('loading ground state --> done')
except FileNotFoundError:
    print('loading ground state --> not found')
    print('computing ground state')
    psi_gs = get_ground_state(steps=50000, 
                              step_size=2/100, 
                              delta=delta_i, 
                              omega=omega_l_i)
    print('computing ground state --> done')
    np.save(directory_path
            + f'/input/ground_state_omega_l_{omega_l_i:.2f}_'
            + f'delta_{delta_i:.2f}_'
            + params_str, psi_gs)

'''
3. TWA-noise ----------------------------------------------------------
'''
noise_k = (np.random.normal(scale=1/np.sqrt(4*n_atoms*dvk), size=(2, nx*nz)) +
           1.0j*np.random.normal(scale=1/np.sqrt(4*n_atoms*dv), size=(2, nx*nz)))

mask = ((kx_**2 + kz_**2) <= (kx_**2 + kz_**2).max() * 4 / 9).reshape(nx*nz)

noise = ifourier_transform(noise_k.reshape((2, nx*nz)) * mask[None, :],
                           axis=-1).reshape(2*nx*nz)

if sample==0:
    noise = 0*noise

'''
4. simulation ---------------------------------------------------------
'''

print(f'gp simulation for sample {sample:.0f} --> started')

t_frames = 201
n_steps = simulation_time // 0.01
dt = simulation_time / n_steps

psi = rk4(fun=sequence_ramp,
          y0=psi_gs + noise,
          frames=t_frames,
          steps=n_steps,
          step_size=dt,
          delta_fun=detuning_ramp,
          lattice_fun=lattice_ramp)

print(f'gp simulation for sample {sample:.0f}--> done')

'''
5. save data ----------------------------------------------------------
'''
print('saving data')
n = ((psi.conj()*psi).real.reshape((2, nx*nz, t_frames)) * 
     dv[None, :, None]).sum(axis=1)

polarization = (n[0] - n[1]) / (n[0] + n[1])

np.save(out_file_name, psi)

