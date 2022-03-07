"""SMACM Model Integrator

Adam M. Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
3.2.2022

This code contains functions that integrate the model equations of SMACM.
Each function that requires computation is integrated with numba, making each 
computation significantly faster than it otherwise would be.
"""

import numpy as np 
from numba import jit

@jit(nopython=True)
def integrate_SMACM(N_seconds, ics, P_ts, F_ts, Td_ts, model_params):
    """Numerically integrate a set of equations using Newton's method.

    Parameters
    ----------
    t: time
        An array of time values for which the equations will be integrated
    loc: Location subclass
        Location subclass containing model parameters
    ics: list
        List of initial conditions of temperature and moisture (in that order)
    P_ts: numpy array
        Precipitation time series forcing
    F_ts: numpy array
        Solar radiative forcing time series 
    Td_ts: numpy array
        Time series of dew point temperature forcing
    
    Returns
    -------
    T_ts: numpy array
        Temperature time series
    m_ts: numpy array
        moisture time series 
    """
    T_ts = np.zeros_like(F_ts)
    m_ts = np.zeros_like(F_ts)

    T_ts[:, 0], m_ts[:, 0] = ics

    for sec in range(1, N_seconds):
        T_ts[:, sec] = T_ts[:, sec - 1] + T_flux(T_ts[:, sec - 1], m_ts[:, sec - 1], F_ts[:, sec - 1], Td_ts[:, sec - 1], model_params)
        m_ts[:, sec] = m_ts[:, sec - 1] + m_flux(T_ts[:, sec - 1], m_ts[:, sec - 1], P_ts[sec - 1], Td_ts[:, sec - 1], model_params) 

    return T_ts, m_ts

@jit(nopython=True)
def T_flux(T, m, F, Td, model_params):
    """Get temperature flux from SMACM

    Parameters
    ----------
    T: float
        temperature at time of evaluation
    m: float
        soil moisture fraction at time of evaluation
    F: float
        radiative forcing at time of evaluation
    Td: float
        dew point temperature at time of evaluation
    model_params: list
        list of calibrated parameters from model
    
    Returns
    -------
    flux: float
        Value of flux in temperature equation of SMACM
    """
    alpha_s, alpha_r, L, gamma, nu, m_0, C, mu = model_params

    feedback = alpha_s + alpha_r + L * gamma * nu * (m + m_0)
    temp_diff = T - Td
    flux = C**(-1) * (F - feedback * temp_diff)
    return flux

@jit(nopython=True)
def m_flux(T, m, P, Td, model_params):
    """Get moisture flux from SMACM

    Parameters
    ----------
    T: float
        temperature at time of evaluation
    m: float
        soil moisture fraction at time of evaluation
    P: float
        precipitation forcing at time of evaluation
    Td: float
        dew point temperature at time of evaluation
    model_params: list  
        list of calibrated parameters from model
    
    Returns
    -------
    flux: float
        Value of flux in moisture equation of SMACM
    """
    alpha_s, alpha_r, L, gamma, nu, m_0, C, mu = model_params

    temp_diff = T - Td
    flux = mu**(-1) * (P - temp_diff * nu * m * gamma)
    return flux 