# -*- coding: utf-8 -*-                                                         
"""                                                                             
III. EXECUTE SIMULATION
GP Simulation
geometry: 3D-axial
system: 3-component BEC
author: Federico Serrano
Physics and Astronomy Department
Washington State University
"""

import numpy as np
import pandas as pd
from system_functions import *

# path where 'execute_simulation.py' is stored
# directory_path = os.path.dirname(os.path.realpath(__file__))

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
ramp_time_ms = 50
ramp_time = ramp_time_ms / 1000 / t_unit

simulation_time = ramp_time
# simulation_time = (ramp_time + 12 * np.sqrt(ramp_time))

delta_i = 5000 / f_recoil
delta_f = -1500 / f_recoil

omega_i = 0.2
omega_f = 0.2

sample = 0

def detuning_ramp(t):
    delta = ((delta_i + (delta_f - delta_i)*t/ramp_time)*(t<=ramp_time)
             + delta_f*(t>ramp_time))
    return delta

def lattice_ramp(t):
    omega = ((omega_i + (omega_f - omega_i)*t/ramp_time)*(t<=ramp_time)
             + omega_f*(t>ramp_time))
    return omega

sample_str = (f'sample_{sample:.0f}_ramp_time_{ramp_time_ms:.0f}ms_'
              + f'delta_i_{delta_i*f_recoil:.0f}Hz_'
              + f'delta_f_{delta_f*f_recoil:.0f}Hz_'
              + f'omega_i_{omega_i:.2f}_omega_f_{omega_f:.2f}_')

out_file_name = (directory_path +
                 '/output/wavefunction_' + sample_str + params_str)

'''
2. get ground state ---------------------------------------------------
'''

print('loading ground state')
try:
    psi_gs = np.load(directory_path 
                     + f'/input/ground_state_omega_l_{omega_i:.2f}_'
                     + f'delta_{delta_i:.2f}_'
                     + params_str)
    print('loading ground state --> done')
except FileNotFoundError:
    print('loading ground state --> not found')
    print('computing ground state')
    psi_gs = get_ground_state(steps=400000, 
                              step_size=4/100,
                              delta=delta_i,
                              omega=omega_i)
    print('computing ground state --> done')
    np.save(directory_path                                                      
            + f'/input/ground_state_omega_l_{omega_i:.2f}_'                       
            + f'delta_{delta_i:.2f}_'                                            
            + params_str, psi_gs)

'''
3. TWA-noise ----------------------------------------------------------
'''
noise = (np.random.normal(scale=1/np.sqrt(4*n_atoms*dz), 
                          size=3*nz) +
         1j*np.random.normal(scale=1/np.sqrt(4*n_atoms*dz),
                             size=3*nz))

if sample==0:
    noise = 0*noise

'''
4. simulation ---------------------------------------------------------
'''

print(f'gp simulation for sample {sample:.0f} --> started')

t_frames = 201
n_steps = simulation_time // 0.005
dt = simulation_time / n_steps

assert dt < (1 / (U.max() + (kz**2).max())) / 5

psi = rk4(fun=sequence_ramp,
          y0=psi_gs + noise,
          frames=t_frames,
          steps=n_steps,
          step_size=dt,
          delta_fun=detuning_ramp,
          lattice_fun=lattice_ramp)

print(f'gp simulation for sample {sample:.0f}--> done')

np.save(out_file_name, psi)
