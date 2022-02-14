"""
Adam M. Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu

A simple Heaviside function that contributes to verifying the analytical
solution to the soil mositure time series. 
"""

import numpy as np

def getHeaviside(size, where):
    """
    Simple heaviside function.

    Args:
        size: int
            size of heaviside array

        where: int
            where does the heaviside step occur?
    """
    
    """
    if t < where, tmp = 0. else, tmp = 1.
    """
    tmp = np.zeros(size)
    for i in range(0, tmp.shape[0]):
        if i < where:
            continue 
        else:
            tmp[i] = 1

    return tmp
