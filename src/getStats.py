"""
Adam M. Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu

Set of functions that calculate statistical quantities of an array.
"""
import numpy as np 

def getNMoment(n, array):
    """
    Function which returns the Nth non-central moment of a time series/set of
    points.

    Args:
        n: int
            Nth moment we are calculating

        array: ndarray
            Data to have Nth moment to be taken
    """

    """
    \bar{x}^{n} = \sum x^{n}
    """
    narray = array**n
    summed = np.sum(narray)
    N_elements = np.shape(array)[0]
    n_moment = summed / N_elements
    
    return n_moment

def getZeta(n, param_list, precip_array):
    """
    In 'On the influence of soil moisture on temperature extremes,' a quantity
    \zeta_{n} is calculated that determines the nth central moment of the soil
    moisture distribution. 

    Args:
        n: int
            the n, such that \zeta_{n}

        param_list: list
            a list of relevant parameters

        precip_array: ndarray
            precip array time series (which zeta_{n}) relies on
    """

    """
    Calc n and n+1 moment of precip TS
    """
    p_mom_n = getNMoment(n, precip_array)
    p_mom_n1 = getNMoment(n + 1, precip_array)
    
    """
    Extract parameter values 
    """
    C = param_list[0]
    F = param_list[1]
    alpha = param_list[2]
    v_L = param_list[3]
    L = param_list[4]
    gamma = param_list[5]
    T_D = param_list[6]
    mu = param_list[7]
    
    """
    Calc time scale and \kappa parameter
    """
    tau = mu * alpha * (F * v_L * gamma)**(-1)
    kappa = L * v_L * gamma * alpha**(-1)
    
    """
    Calc zeta_{n}
    """
    zeta = tau * kappa * p_mom_n1 * (n+1)**(-1) + tau * p_mom_n * n**(-1)
    
    return zeta
    
