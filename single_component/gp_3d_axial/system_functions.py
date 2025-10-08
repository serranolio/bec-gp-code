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

'''
2. hamiltonian ---------------------------------------------------------
'''

def H_single_particle(psi, detuning, lattice_strength):
    original_shape = psi.shape

    # precompute potential term
    lattice = (2 * lattice_strength * np.sin(k_l*z_)**2).reshape(nx*nz)
    potential_term = (U + lattice) * psi

    # apply hamiltonian
    H_psi = (T(psi) - 2 * K_z(psi)
             + (1 - detuning / 2) * psi) + potential_term


    return np.array(H_psi).reshape(original_shape)

def H_interaction(psi):
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
             + H_interaction(psi))
    
    chemical_potential = ((psi.conj() * H_psi) * dv).sum() / norm
    
    return -H_psi + chemical_potential * psi

def sequence_ramp(t, psi, delta_fun, lattice_fun):
    omega_t = lattice_fun(t)
    delta_t = delta_fun(t)

    H_psi = (H_single_particle(psi, 
                               detuning=delta_t, 
                               lattice_strength=omega_t)
             + H_interaction(psi))
    return -1j*H_psi

'''
4. Ground state -------------------------------------------------------
'''
def get_ground_state(steps, step_size, delta, omega):
    if g==0:
        psi0 = (1/pi)**(1/4)*np.exp(-(x_**2 + z_**2)/2) + 0j
        psi0 = psi0.reshape((nx*nz, ))
    else:
        trap = ((wx*x_)**2 + (wz*z_)**2)/4
        psi0 = (np.sqrt(np.abs(mu - trap)/g/n_atoms) + 0j)*(trap < mu)
        psi0.shape = (nx*nz, )
        norm = ((dv*psi0*psi0.conj()).real).sum()
        psi0 = psi0 / np.sqrt(norm)

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
5. TWA noise generator
'''

def get_noise(psi_gs, energy_cutoff=None):
    # define noise wavefunction
    dpsi = np.zeros_like(psi_gs, dtype=complex)
    dvk_reshaped = dvk.reshape((nx, nz))
    density = (np.abs(psi_gs)**2)
    #psi_sq_reshaped = psi_sq.reshape((3, nx * nz))

    for i, q_z in enumerate(kz[:nz]):
        for j, q_x in enumerate(kx):
            coefficient = (1 if not energy_cutoff or (q_x**2 + q_z**2) <=
                           energy_cutoff else 0)

            correlation = 1 / (4 * n_atoms * dvk_reshaped[j, i])

            # Random complex coefficients
            alpha = (np.random.normal(scale=correlation, size=1) +
                     1j * np.random.normal(scale=correlation, size=1))

            epsilon_0 = q_z**2 + q_x**2
            gn_ = g * n_atoms * density
            
            xi_q = epsilon_0 + gn_
            eps_q = np.sqrt(xi_q**2 - gn_**2)

            u = np.sqrt(0.5 * (xi_q / eps_q + 1))
            v = np.sqrt(0.5 * (xi_q / eps_q - 1))

            #u = np.sqrt(np.abs((epsilon_k**2 + gn_) / 
            #                   (2 * np.sqrt(epsilon_k * 
            #                                (epsilon_k + 2 * gn_)))
            #                   - 0.5))
            #v = np.sqrt(u**2 - 1)

            plane_wave_1 = (np.exp(1j * q_z * z_) * 
                            jv(0, q_x * x_)).reshape((nx * nz))
            plane_wave_2 = (np.exp(-1j * q_z * z_) * 
                            jv(0, q_x * x_)).reshape((nx * nz))

            dpsi += coefficient * (alpha * (u * plane_wave_1) +
                                   alpha.conj() * (v * plane_wave_2))

    return dpsi

'''
6. fourier transform --------------------------------------------------
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
