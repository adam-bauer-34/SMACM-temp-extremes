"""
Adam M. Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu

Simple Newtonian integrator.
"""

import numpy as np

def getSimulate(N_steps, step_size, variable, flux, fluxargs):
    """
    Simple Newtonian integrator. 

    Args:
        N_steps: int
            Number of steps integration will take.

        step_size: float
            Size of steps in the x (or t, or whatever) dimension

        variable: ndarray
            variable being integrated

        flux: func
            flux function

        fluxargs: list
            arguments for flux function
    """
    for i in range(1, N_steps):
        variable[i] = variable[i-1] + step_size * flux(variable[i-1], fluxargs)
    
    return variable
