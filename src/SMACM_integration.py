"""SMACM Model Integrator

Adam M. Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
3.2.2022

This code contains functions that integrate the model equations of SMACM.
Each function that requires computation is integrated with numba, making each 
computation significantly faster than it otherwise would be.
"""

import gc 

import numpy as np 
from numba import jit
import matplotlib.pyplot as plt

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
def integrate_SMACM_chunked(N_seconds, ics, P_ts, F_ts, Td_ts, model_params):
    """Numerically integrate a set of equations using Newton's method with chunking.

    Chunking here will hopefully speed up runtime by clearing memory space in the computer during the 
    calculation. We define a "chunk" by a day's worth of time. 

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
    N_sims = len(F_ts)

    T_ts = np.zeros_like(F_ts)
    m_ts = np.zeros_like(F_ts)

    secs_in_day = 86400
    N_sec_integrated = 0

    while N_sec_integrated < N_seconds:
        T_ts_chunk = np.zeros((N_sims, secs_in_day + 1))
        m_ts_chunk = np.zeros_like(T_ts_chunk)

        tmp_LB = N_sec_integrated
        tmp_UB = N_sec_integrated + secs_in_day
        
        if N_sec_integrated == 0:
            T_ts_chunk[:, 0], m_ts_chunk[:, 0] = ics
        else:
            T_ts_chunk[:, 0], m_ts_chunk[:, 0] = T_ts[:, tmp_LB], m_ts[:, tmp_LB]

        if N_sec_integrated != N_seconds - secs_in_day:
            for sec in range(1, secs_in_day + 1):
                T_ts_chunk[:, sec] = T_ts_chunk[:, sec - 1] + T_flux(T_ts_chunk[:, sec - 1], m_ts_chunk[:, sec - 1], F_ts[:, sec - 1], Td_ts[:, sec - 1], model_params)
                m_ts_chunk[:, sec] = m_ts_chunk[:, sec - 1] + m_flux(T_ts_chunk[:, sec - 1], m_ts_chunk[:, sec - 1], P_ts[sec - 1], Td_ts[:, sec - 1], model_params)

            T_ts[:, tmp_LB:tmp_UB + 1] = T_ts_chunk
            m_ts[:, tmp_LB:tmp_UB + 1] = m_ts_chunk
        
        else:
            for sec in range(1, secs_in_day):
                T_ts_chunk[:, sec] = T_ts_chunk[:, sec - 1] + T_flux(T_ts_chunk[:, sec - 1], m_ts_chunk[:, sec - 1], F_ts[:, sec - 1], Td_ts[:, sec - 1], model_params)
                m_ts_chunk[:, sec] = m_ts_chunk[:, sec - 1] + m_flux(T_ts_chunk[:, sec - 1], m_ts_chunk[:, sec - 1], P_ts[sec - 1], Td_ts[:, sec - 1], model_params)


            T_ts[:, tmp_LB:tmp_UB] = T_ts_chunk[:, :-1]
            m_ts[:, tmp_LB:tmp_UB] = m_ts_chunk[:, :-1]

        #del T_ts_chunk, m_ts_chunk
        #gc.collect()

        N_sec_integrated += secs_in_day

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