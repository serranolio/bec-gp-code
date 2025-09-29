# -*- coding: utf-8 -*-
"""
II. SYSTEM FUNCTIONS
GP simulation
geometry: 1D
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
1. split-step-fourier
'''
def split_step_imag_time(psi0, 
                         steps, 
                         step_size,
                         delta,
                         omega,
                         normalize_each_step=True):
    psi = psi0.copy()
    shape0 = psi.shape
    psi = psi.reshape((3, -1))
    dt = step_size
    
    for _ in range(steps):
        # --- Compute nonlinear interaction ---
        gn = g_1d * (psi * psi.conj()).real.sum(axis=0)

        # --- Real space half-step: potential + interaction + spin coupling ---
        V0 = (-delta/2) + (U + 2*omega*np.sin(k_l*z)**2 + gn)
        V1 = (delta/2) + (U + 2*omega*np.sin(k_l*z)**2 + gn)
        V2 = (3*delta/2 + 2*q_zeeman) + (U + 2*omega*np.sin(k_l*z)**2 + gn)

        psi0 = psi[0] * np.exp(-0.5 * dt * V0)
        psi1 = psi[1] * np.exp(-0.5 * dt * V1)
        psi2 = psi[2] * np.exp(-0.5 * dt * V2)

        # Apply spin coupling (first-order split)
        psi0 += -0.5 * dt * omega_r/2 * psi1
        psi1 += -0.5 * dt * omega_r/2 * (psi0 + psi2)
        psi2 += -0.5 * dt * omega_r/2 * psi1

        # --- Kinetic full step: via FFT ---
        psi0_k = np.fft.fft(psi0)
        psi1_k = np.fft.fft(psi1)
        psi2_k = np.fft.fft(psi2)

        psi0_k *= np.exp(-dt * (kz + 1)**2)
        psi1_k *= np.exp(-dt * (kz - 1)**2)
        psi2_k *= np.exp(-dt * (kz - 3)**2)

        psi0 = np.fft.ifft(psi0_k)
        psi1 = np.fft.ifft(psi1_k)
        psi2 = np.fft.ifft(psi2_k)

        # --- Real space half-step again ---
        psi0 *= np.exp(-0.5 * dt * V0)
        psi1 *= np.exp(-0.5 * dt * V1)
        psi2 *= np.exp(-0.5 * dt * V2)

        psi0 += -0.5 * dt * omega_r/2 * psi1
        psi1 += -0.5 * dt * omega_r/2 * (psi0 + psi2)
        psi2 += -0.5 * dt * omega_r/2 * psi1

        # --- Renormalize ---
        if normalize_each_step:
            norm = dz * (np.abs(psi0)**2 + np.abs(psi1)**2 + np.abs(psi2)**2).sum()
            psi0 /= np.sqrt(norm)
            psi1 /= np.sqrt(norm)
            psi2 /= np.sqrt(norm)

        # Update psi
        psi[0] = psi0
        psi[1] = psi1
        psi[2] = psi2

    return psi.reshape(shape0)


'''
2. kinetic energy operators --------------------------------------------
'''
def T(psi, fk):
    return np.fft.ifft(fk*np.fft.fft(psi, axis=-1), 
                       axis=-1)

'''
3. hamiltonian ---------------------------------------------------------
'''

def H_single_particle(psi, detuning, lattice_strength):
    original_shape = psi.shape
    psi = psi.reshape((3, -1))

    # precompute potential term
    lattice = (2 * lattice_strength * np.sin(k_l*z)**2)
    potential_term = (U + lattice)[None, :] * psi

    # apply hamiltonian
    H_psi = [
        T(psi[0], (kz+1)**2)
        + (- detuning / 2) * psi[0]
        + omega_r / 2 * psi[1],

        T(psi[1], (kz-1)**2)
        + (detuning / 2) * psi[1]
        + omega_r /2 * (psi[0] + psi[2]),

        T(psi[2], (kz-3)**2)
        + (3 * detuning / 2 + 2 * q_zeeman) * psi[2]
        + omega_r / 2 * psi[1]
    ] + potential_term

    return np.array(H_psi).reshape(original_shape)

def H_interaction(psi):
    original_shape = psi.shape
    psi = psi.reshape((3, -1))
    # density potential
    density = np.sum(np.abs(psi)**2, axis=0)
    return g_1d * (density[None, :] * psi).reshape(original_shape)

'''
4. sequence functions --------------------------------------------------
'''

def sequence_cooling(t, psi, delta, omega):
    norm = np.sum(dz*np.abs(psi.reshape((3, -1)))**2)

    H_psi = (H_single_particle(psi,
                               detuning=delta,
                               lattice_strength=omega)
             + H_interaction(psi))

    chemical_potential = ((psi.conj() * H_psi).reshape((3, -1))
                          * dz).sum()/norm

    return -H_psi + chemical_potential*psi

def sequence_ramp(t, psi, delta_fun, lattice_fun):
    omega_t = lattice_fun(t)
    delta_t = delta_fun(t)

    H_psi = (H_single_particle(psi,
                               detuning=delta_t,
                               lattice_strength=omega_t)
             + H_interaction(psi))
    return -1j*H_psi

'''
5. Ground state -------------------------------------------------------
'''
def get_ground_state(steps, step_size, delta, omega):
    if g==0:
        psi0 = (1/pi)**(1/4)*np.exp(-(z**2)/2) + 0j
        psi0 = psi0.reshape((nz, ))
    else:
        trap = ((wz*z)**2)/4
        psi0 = (np.sqrt(np.abs(mu - trap)) + 0j)*(trap < mu)
        norm = ((dz*psi0*psi0.conj()).real).sum()
        psi0 = psi0/np.sqrt(norm)
    psi_initial = np.array([psi0/np.sqrt(3), 
                            psi0/np.sqrt(3), 
                            psi0/np.sqrt(3)]).reshape(3*nz)

    #psi = rk4(fun=sequence_cooling,
    #          y0=psi_initial,
    #          frames=2,
    #          steps=steps,
    #          step_size=step_size,
    #          delta=delta,
    #          omega=omega)[:, -1]
    psi = split_step_imag_time(psi_initial, steps, step_size, delta, omega)
    
    psi.shape = (3, -1)
    norm = (dz*psi.conj()*psi).sum()
    psi = psi/np.sqrt(norm)

    return np.array(psi.reshape(3*nz))

