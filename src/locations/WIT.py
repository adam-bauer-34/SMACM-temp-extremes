"""
Adam M. Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
2.26.2022

This code contains the location class for Witchita, KS,
or "WIT." WIT is a subclass of the Location abstract class. 
"""

import numpy as np 

from .Location import Location

class WIT(Location):
    """
    Class object for Witchita, KS. Subclass of the Location
    abstract class.
    """
    def __init__(self):
        """
        Of use constants.
        """
        self.L = 2.5e6
        self.C = 4180 
        self.P_surf = 101325
        self.R_w = 461.52
        self.T_0 = 273.15 

        """
        Model parameters.
        """
        self.alpha_s = 4
        self.alpha_r = 7
        self.nu = 0.02
        self.mu = 40
        self.m_0 = 0.2
        self.F_mean = 226.44
        self.F_std = 47.42
        self.Td_mean = 291.29
        self.Td_std = 2.92
        self.omega = 3.33
        self.omega_s = self.omega * 86400 # days -> seconds
        self.p_0 = 0.12
        self.p_scale = 1
        self.F_warming_max = None
        self.m_mean = 0.00238528

    @property
    def gamma(self):
        """
        Calculate gamma, the derivative of the Clausius-Clapeyron relationship
        evaluated at the dew point temperature. 
        """
        factor1 = 611 # Pa
        factor2 = 0.622 # -
        exp_arg = self.L * self.R_w**(-1) * (self.T_0**(-1) -
                                             self.Td_mean**(-1))
        prefactor = factor1 * factor2 * self.L * (self.R_w * self.P_surf *
                                                  self.Td_mean**2)**(-1)
        return prefactor * np.exp(exp_arg)

    def calibrate_warming_simulations(self, max_warming):
        """
        Calculate the maximum increase in F needed to warm the mean temperature
        by max_warming.

        Arguments
        ---------
        max_warming: float
            maximum amount of warming to allow in the location (in Kelvin)
        """
        
        """
        Using nullcline equation, we can write the mean temperature as:
        """
        mean_T = self.Td_mean + self.F_mean * (self.alpha_s + self.alpha_r +
                                               self.L * self.gamma * self.nu *
                                               (self.m_0 + self.m_mean))**(-1)
        """
        Then add a forcing anomaly such that the temperature is equal to the
        mean_T + some amount. Solving for the forcing anomaly, we have:
        """
        diff_T = mean_T - self.Td_mean + max_warming 
        feedback = self.alpha_r + self.alpha_s + self.gamma * self.L * self.nu * (self.m_0 + self.m_mean)
        self.F_warming_max = diff_T * feedback - self.F_mean
