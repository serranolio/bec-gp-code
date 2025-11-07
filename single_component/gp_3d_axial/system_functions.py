# -*- coding: utf-8 -*-
"""
II. SYSTEM FUNCTIONS
GP simulation
geometry: 3D-axial
system: 3-component BEC
author: Federico Serrano
Physics and Astronomy Department
Washington State University
""" 

from parameters import *


'''
0. RK4 ----------------------------------------------------------------
'''
def rk4(fun,
        y0,
        frames,
        steps,
        step_size,
        **kwargs):

    t = np.arange(steps)*step_size
    indices = np.linspace(0, steps-1, frames, dtype=int)

    t_eval = t[indices]
    y = np.zeros((len(y0), frames), dtype='complex')
    y[:, 0] = y0.copy()

    dt = step_size
    yi = y0.copy()
    j = 0
    for i in range(len(t)):
        ti = t[i]
        k1 = fun(ti, yi, **kwargs)
        k2 = fun(ti + dt/2, yi + dt/2 * k1, **kwargs)
        k3 = fun(ti + dt/2, yi + dt/2 * k2, **kwargs)
        k4 = fun(ti + dt, yi + dt * k3, **kwargs)

        yi += dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        if i in indices:
            y[:, j] = yi
            j += 1
    return y

'''
1. kinetic energy operators --------------------------------------------
'''
def T(psi):
    shape0 = psi.shape
    shape1 = (nx, nz)
    T_psi = (Tx @ psi.reshape(shape1) +
             (np.fft.ifft(kz**2*np.fft.fft(psi.reshape(shape1), 
                                           axis=-1), 
                          axis=-1))).reshape(shape0)

    return T_psi

def K_z(psi):
    shape0 = psi.shape
    shape1 = (nx, nz)
    Kz_psi = (np.fft.ifft((-kz)*np.fft.fft(psi.reshape(shape1), 
                                           axis=-1), 
                          axis=-1)).reshape(shape0)

    return Kz_psi

def low_energy_projector(psi, cutoff):
    return psi

'''
2. hamiltonian ---------------------------------------------------------
'''

def H_single_particle(psi, detuning, lattice_strength):
    original_shape = psi.shape

    # precompute potential term
    lattice = (2 * lattice_strength * np.sin(k_l*z_)**2).reshape(nx*nz)
    potential_term = (U + lattice) * psi

    # apply hamiltonian
    #H_psi = (T(psi) - 2 * K_z(psi)
    #         + (1 - detuning / 2) * psi) + potential_term
    H_psi = (T(psi)
             + (-detuning / 2) * psi) + potential_term


    return np.array(H_psi).reshape(original_shape)

def H_interaction(psi, g):
    original_shape = psi.shape

    # density potential
    density = np.abs(psi)**2
    return g * n_atoms * (density * psi).reshape(original_shape)

'''
3. sequence functions --------------------------------------------------
'''

def sequence_cooling(t, psi, delta, omega):
    norm = np.sum(dv*np.abs(psi)**2)

    H_psi = (H_single_particle(psi, 
                               detuning=delta, 
                               lattice_strength=omega)
             + H_interaction(psi, g_eff))
    
    chemical_potential = ((psi.conj() * H_psi) * dv).sum() / norm
    
    return -H_psi + chemical_potential * psi

def sequence_ramp(t, psi, delta_fun, lattice_fun, g):
    omega_t = lattice_fun(t)
    delta_t = delta_fun(t)

    H_psi = (H_single_particle(psi, 
                               detuning=delta_t, 
                               lattice_strength=omega_t)
             + H_interaction(psi, g_eff))
    return -1j*H_psi


'''
4. Ground state -------------------------------------------------------
'''
def get_ground_state(steps, step_size, delta, omega):
    if g_eff==0:
        psi0 = (1/pi)**(1/4)*np.exp(-(x_**2 + z_**2)/2) + 0j
        psi0 = psi0.reshape((nx*nz, ))
    else:
        trap = ((wx*x_)**2 + (wz*z_)**2)/4
        psi0 = (np.sqrt(np.abs(mu - trap)/g_eff/n_atoms) + 0j)*(trap < mu)
        psi0.shape = (nx*nz, )
        norm = (dv*np.abs(psi0)**2).sum()
        psi0 = psi0 * np.exp(-1j * k_l * z_.reshape(nx*nz)) / np.sqrt(norm)

    psi = rk4(fun=sequence_cooling,
              y0=psi0,
              frames=2,
              steps=steps,
              step_size=step_size,
              delta=delta,
              omega=omega)[:, -1]
    
    norm = (dv*psi.conj()*psi).sum()
    psi = psi / np.sqrt(norm)

    return np.array(psi.reshape(nx*nz))

'''
5. fourier transform --------------------------------------------------
'''

def fourier_transform(state, axis=-1):
    state = np.swapaxes(state, axis, -1)
    shape0 = state.shape
    shape1 = shape0[:-1] + (nx, nz)

    transform = lx**2*dht.dht(state.reshape(shape1), axis=-2)
    transform = (dz/np.sqrt(2*pi))*np.fft.fft(transform, axis=-1)
    transform = np.fft.fftshift(transform, axes=-1)
    transform = np.swapaxes(transform.reshape(shape0), axis, -1)
    return transform

def ifourier_transform(state, axis=-1):
    state = np.swapaxes(state, axis, -1)
    shape0 = state.shape
    shape1 = shape0[:-1] + (nx, nz)

    transform = dht.idht(state.reshape(shape1), axis=-2)/lx**2
    transform = (np.sqrt(2*pi)/dz)*np.fft.ifft(transform, axis=-1)
    transform = np.swapaxes(transform.reshape(shape0), axis, -1)
    return transform
