"""
Adam M. Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
2.25.2022

This code contains classes for each loaction for which we have model parameters
fitted using ERA5. Each location class must be of the form of the abstract
class Location, which controls what attributes/methods must be present in each
location.
"""

import numpy as np
from abc import ABC, abstractmethod, abstractproperty

class Location(ABC):
    """
    Abstract location class for the Heatwave-freq simulation.

    Attributes
    ----------
    self.L: float
        latent enthalpy of liquid water (J kg H20^{-1})
    self.C: float
        effective heat capacity of land surface (J  K^{-1})
    self.P_surf: float
        surface pressure (Pa)
    self.R_w: float
        specific gas constant of water (kg H20 J^{-1})
    self.T_0: float
        reference temperature for CC relationship (K)
    alpha_s: float
        Dry feedback parameter associated with sensible heat flux (W K^{-1})
    alpha_r: float
        Dry feedback parameter associated with longwave heat flux (W K^{-1})
    nu: float
        density of near surface air / surface resistance (kg air s m^{-2})
    mu: float
        maximum water holding capacity (kg H20 m^{-2})
    m_0: float
        deep soil moisture fraction (-)
    F_mean: float
        daily mean incoming shortwave radiation (W m^{-2})
    F_std: float
        standard deviation of daily incoming shortwave radiation (W m^{-2})
    Td_mean: float
        daily mean 2m dew point temperature (K)
    Td_std: float
        standard deviation of daily 2m dew point temperature (K)
    gamma: float (property)
        derivative of the Clausius Clapeyron relation eval'd at the mean dew
        point temperature (K^{-1})
    omega: float
        average interval between rain events (days)
    omega_s: float
        average interval between rain events (seconds)
    p_0: float
        average precip event intensity (-)
    p_scale: float
        scale of precip intensity distribution (-)
    F_warming_max: float
        maximum F increase required for ~5 K warming
    m_mean: float
        mean soil moisture after baseline simulation 
        calculated in HWFreqAnalysis.ipynb
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
        
        self.alpha_s = None
        self.alpha_r = None
        self.nu = None
        self.mu = None
        self.m_0 = None
        self.F_mean = None
        self.F_std = None
        self.Td_mean = None
        self.Td_std = None
        self.omega = None
        self.omega_s = None
        self.p_0 = None
        self.p_scale = None
        self.F_warming_max = None
        self.m_mean = None

    @abstractproperty
    def gamma(self):
        pass 

    @abstractmethod
    def calibrate_warming_simulations(self, max_warming):
        pass