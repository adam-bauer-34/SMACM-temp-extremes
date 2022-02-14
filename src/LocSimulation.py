"""
Adam Michael Bauer
adammb4@illinois.edu

This code is a class object which carries out a numerical integration of the 
model equations in SMACM. It's purpose is to integrate the model equations 
in numerous different scenarios, each with an increasing mean dew point 
temperature. It then calculates a baseline percentile for low soil moisture 
(5th percentile in daily mean soil moisture), high daily mean temperature (95th 
percentile), and daily max temperature (95th percentile).

In each of the remaining simulations with higher mean dew point temperature, the 
percent of days which exceed these baselines are calculated. This is intended
to show that more "very dry" or "very warm" days happen in a global warming situation. 

This code was created in support of a manuscript, with citation 
Bauer, A. M. et al., On the impact of soil moisture on temperature extremes, in prep., 2022.
"""

import numpy as np 
import xarray as xr 
import datetime

class LocSimulation:
    """
    class: LocSimulation

    Simulates for a given location the increase in extreme dry events for a warming event
    of +5 K globally.

    args:
    loc_string: string
        location identifier string. to be used in filenames of saved outputs.

    param_dict: dictionary 
        contains parameter values:
            alpha_s:    dry radiative feedback from sensible heat flux 
            alpha_r:    dry radiative feedback from downwards LW radiation
            nu:         model parameter
            m_0:        deep soil moisture content
            F_mean:     mean daily solar radiation
            F_std:      standard deviation of daily solar radiation
            Td_mean:    mean daily mean dew point temperature
            Td_std:     std daily mean dew point temperature

    N_summers: int
        number of summers to simulate (higher N_summers, better data)

    N_simulations: int
        number of increments 

    import_precip: bool
        are we importing a previously made precip time series? 
"""

    def __init__(self, loc_string, param_dict, N_simulations, N_summers, import_precip):
        self.loc_string = loc_string
        self.param_dict = param_dict
        self.N_summers = N_summers
        self.N_simulations = N_simulations
        self.import_precip = import_precip

        # make base filenames and path
        self.path ="/data/keeling/a/adammb4/heatwaves-physics/data/heatwave-freq-data/"

        current_date = datetime.datetime.now()
        year = str(current_date.year)
        day = str(current_date.day)
        month = str(current_date.month)
        self.base_filename = month + '-' + day + '-' + year + '-' + self.loc_string + "-"

        # of use constants
        self.L = 2.5e6 # J / kg H20 
        self.C = 4180 # J/K 
        self.P_surf = 101325 # Pa
        self.R_w = 461.52 # J / kg 
        self.T_0 = 273.15 # K 
        self.s_in_day = 86400 # s / day
        self.maxT_departure = 5 # K, maximum amount dew point temperature inc

        # extract parameters 
        self.alpha_s = self.param_dict['alpha_s'] # dry feedback assoc. sensible heat flux
        self.alpha_r = self.param_dict['alpha_r'] # dry feedback assoc. lw radiative 
        self.alpha = self.alpha_s + self.alpha_r # total dry feedback 
        self.nu = self.param_dict['nu'] # near surface air density / surface resistance 
        self.mu = self.param_dict['mu'] # water capacity 
        self.m_0 = self.param_dict['m_0'] # deep soil capacity
        self.F_mean = self.param_dict['F_mean'] # mean daily sw radiative forcing 
        self.F_std = self.param_dict['F_std'] # std daily sw radiative forcing 
        self.Td_mean = self.param_dict['Td_mean'] # mean near surface daily dew point 
        self.Td_std = self.param_dict['Td_std'] # std near surface daily dew point 
        self.omega = self.param_dict['omega'] # in days / event (i.e., an event happens every omega days)
        self.omega_s = self.omega * self.s_in_day # seconds / event
        self.p_0 = self.param_dict['p_0'] # precip gamma dist shape param  
        self.p_scale = self.param_dict['p_scale'] # precip gamma dist scale param

        # number of days in N summers
        self.N_days = self.N_summers * 90 # 90 days in summer 
        self.N_seconds = self.N_days * self.s_in_day # seconds in a summer 
        self.time = np.arange(0, self.N_seconds, 1) # make time in seconds total in simulation
        
        # if making new precip time series, make it. if not, import it.
        if self.import_precip == False:
            print("Making new precip forcing...")
            self.makePrecipForcingTS()
        
        else:
            print("Importing precip time series...")
            self.importPrecip()

        # make T_d range
        self.makeTdMeans()

        # make gamma with Td_means 
        self.makeMeanGammas()

        # make mean taus and Zs 
        self.taus = self.mu * self.F_mean**(-1) * (self.alpha * (self.nu * self.gamma_means)**(-1) + self.m_0 * self.L) 
        self.Z_means = self.omega_s**(-1) * self.taus

        # make forcings 
        self.makeDailyMeanForcingDists() # make daily mean forcing distributions for F and T_d 

        # make time series  
        self.T_ts = np.zeros((self.N_simulations, self.N_seconds))
        self.T_ts[:, 0] = self.Td_means
        self.m_ts = np.zeros_like(self.T_ts)

         # make daily maximum temp and daily mean t & m 
        self.T_dailymax = np.zeros((self.N_simulations, self.N_days)) 
        self.T_dailymean = np.zeros_like(self.T_dailymax)
        self.m_dailymean = np.zeros_like(self.T_dailymax)

        # make array for exceedences 
        self.Tmax_exceedences = np.zeros(self.N_simulations)
        self.Tdaily_exceedences = np.zeros_like(self.Tmax_exceedences)
        self.mdaily_exceedences = np.zeros_like(self.Tmax_exceedences)

        print("Location Simulation object ready!")

    def makePrecipForcingTS(self):
        """
        Create a precipitation time series for other simulations. Each precip
        event will be drawn from a Gamma distribution with shape parameter
        p_0 and scale parameter p_scale. The events occur at Poisson intervals.

        Outputs:
            P_ts: precipitation time series
            N_events: number of precipitation events which occur in the time series.
        """
        self.P_ts = np.zeros(int(self.N_seconds))

        # make precip distributions for experimental runs 
        sec = 0
        freq_tracker = 0
        self.N_events = 0
        while sec <= self.N_seconds:
            if sec == freq_tracker:
                self.P_ts[sec] = np.random.gamma(self.p_0, self.p_scale) # select precip event magnitude from gamma distribution 
                freq_tracker += int(np.random.poisson(self.omega_s)) # the next event occurs freq_tracker + ~omega seconds later 
                self.N_events += 1

            sec += 1
        
        print("Saving precipitation time series...")
        precip_ds = xr.Dataset(data_vars={"precip": (["time"], self.P_ts),
                                         },
                               coords={"time": (["time"], self.time),}
                              )
        precip_ds.to_netcdf(path=self.path + "precip_ts_" + str(self.N_summers)
                            + "sum.nc",
                           mode="w", format="NETCDF4", engine="netcdf4")
        print("Done!")

    def importPrecip(self):
        """
        Import precip time series to force the model equations.

        Args:
            date_and_locstring:
                the date of the precip ts you want to import and its locstring.
                i.e., "2-3-2022-SGP"
        Output:
            self.P_ts: precip forcing time series
        """
        # open precip netcdf
        precip_filename = self.path + "precip_ts_" + str(self.N_summers) + "sum.nc"
        precip_ds = xr.open_dataset(precip_filename)
        
        # extract precip time series
        self.P_ts = precip_ds["precip"].values 
        print("Precip time series successfully imported!")

    def makeTdMeans(self):
        """
        Create a list of dew point temperature means 
        that increment to our max warming situation, which is 
        a global temperature increase of 5 K

        Output: 
            - self.Td_means: a list of dew point temperatures, incremented by 5 / N_simulations
        """
        print("Getting ready...")
        self.Td_means = np.zeros(self.N_simulations)
        increment = self.maxT_departure * self.N_simulations**(-1)
        for sim in range(0, self.N_simulations):
            self.Td_means[sim] = self.Td_mean + sim * increment

    def makeMeanGammas(self):
        """
        Create a list of mean gamma values for the increasing dew point temperature
        in each simulation.

        Output:
            - gamma_means: an array of gamma values for the incremeneted dew point temperatures 
        """
        print("Hold on...")
        factor1 = 6.11 * 100 # in pascals
        factor2 = 0.622
        exp_arg = self.L * self.R_w**(-1) * (self.T_0**(-1) - self.Td_means**(-1))
        prefactor = factor1 * factor2 * self.L * (self.R_w * self.P_surf * self.Td_means**2)**(-1)
        self.gamma_means = prefactor * np.exp(exp_arg) 

    def makeDailyMeanForcingDists(self):
        """
        Make daily mean distributions for radiative forcing and dew point temperature.

        Outputs:
            - F_dists: A list of values, for each mean dew point temperture, of the daily mean 
            radiative forcing
            - Td_dists: A list of values, for each mean dew point temperature, of the daily mean
            dew point temperature 
        """
        print("One more thing...")
        self.F_dists = np.zeros((self.N_simulations, self.N_days))
        self.Td_dists = np.zeros((self.N_simulations, self.N_days))
        for simulation in range(0, self.N_simulations):
            self.F_dists[simulation, :] = np.random.normal(self.F_mean, self.F_std, self.N_days)
            self.Td_dists[simulation, :] = np.random.normal(self.Td_means[simulation], self.Td_std, self.N_days)

    def makeModelForcings(self):
        """
        Takes daily means and makes them into second-by-second time series. 

        Output:
            - F_ts: radiative forcing in time series form 
            - Td_ts: dew point temperature in time series form 
        """
        print("Making model forcings...")
        self.F_ts = np.zeros((self.N_simulations, self.N_seconds))
        self.Td_ts = np.zeros((self.N_simulations, self.N_seconds))

        # make F and Td dists
        for simulation in range(0, self.N_simulations):
            for day in range(0, self.N_days):
                self.F_ts[simulation, int(day * self.s_in_day):int((day + 1) * self.s_in_day)] = self.F_dists[simulation, day] # set a day's worth of values to daily mean 
                self.Td_ts[simulation, int(day * self.s_in_day):int((day + 1) * self.s_in_day)] = self.Td_dists[simulation, day] # set a day's worth of values to daily mean

        print("Finished!")


    def makeForcedTimeSeries(self):
        """
        Carry out Newtonian integration of model equations to get time series for 
        soil moisture and temperature.

        Outputs:
            - T_ts: temperature time series 
            - m_ts: moisture time series 
        """
        print("Creating time series... (this could take a bit of time)")
        for sec in range(1, self.N_seconds):
            self.T_ts[:, sec] = self.T_ts[:, sec - 1] + self.getTFlux(self.F_ts[:, sec - 1], self.Td_ts[:, sec - 1], self.T_ts[:, sec - 1], self.m_ts[:, sec - 1], self.gamma_means)
            self.m_ts[:, sec] = self.m_ts[:, sec - 1] + self.getMFlux(self.P_ts[sec - 1], self.Td_ts[:, sec - 1], self.T_ts[:, sec - 1], self.m_ts[:, sec - 1], self.gamma_means) 
        
        print("Finished!")

    def getTFlux(self, F, Td, T, m, gamma):
        """
        Calculate T flux in SMACM.

        F: radiative forcing (W / m^2)
        Td: dew point temperature (K)
        T: temperature (K)
        m: soil moisture (in fractional units)
        gamma: derivative of clausius clapeyron relationship eval'd at mean dew point temp (K^-1)

        Output:
            - returns T_flux
        """
        feedback = self.alpha + self.L * gamma * self.nu * (m + self.m_0)
        temp_diff = T - Td
        T_flux = self.C**(-1) * (F - feedback * temp_diff)
        return T_flux 

    def getMFlux(self, P, Td, T, m, gamma):
        """
        Calculate m flux in SMACM.

        P: precip forcing (fractional units)
        Td: dew point temperature (K)
        T: temperature (K)
        m: soil moisture (in fractional units)
        gamma: derivative of clausius clapeyron relationship eval'd at mean dew point temp (K^-1)

        Output:
            - returns m_flux
        """
        temp_diff = T - Td
        m_flux = self.mu**(-1) * (P - temp_diff * self.nu * m * gamma)
        return m_flux 

    def makeExceedences(self, save_output, dynamic_baseline):
        """
        Calculate the percentage of days which exceed the 95th percentile of
        the baseline daily mean and daily max temperature,
        as well as the percentage of days which are below the 5th percentile of
        the baseline daily mean soil moisture.

        Args:
            save_output: bool
            Should we save the output? If True, then a netCDF will be made with
            Z_means, Tmax, Tdaily, and mdaily exceedences recorded
            
            dynamic_baseline: bool
            Should we shift the baseline for temperature in correspondence with
            how the mean dew point temperature changes? 

        Output:
            - Tmax_exceedences: percentage of days where max daily temperature,
            in incremented simulation, exceed baseline 95th percentile max
            temperature
            - Tdaily_exceedences: percentage of days where daily mean
            temperature, in incremented simulation, exceeds baseline 95th
            percentile daily mean temperature 
            - mdaily_exceedences: percentage of days where daily mean soil
            moisture, in incremented simulation, falls below the 5th percentile
            daily mean soil moisture 
        """
        self.makeDailyMaximums() # make daily maximum array
        self.makeDailyMeans() # make daily mean arrays

        print("Calculating exceedences...")
        baseline_Tmax = np.percentile(self.T_dailymax[0, :], 95) # make baseline 95th percentile for daily ax temp
        baseline_Tdaily = np.percentile(self.T_dailymean[0, :], 95) # make baseline 95th percentile for daily mean temp
        baseline_mdaily = np.percentile(self.m_dailymean[0, :], 5) # make baseline 5th percentile for daily mean soil moisture
        if dynamic_baseline == True:
            increment = self.maxT_departure * self.N_simulations**(-1)
            ex_filename = self.path + self.base_filename + "dyn_exceedences.nc"
        else:
            increment = 0 # if doing static baseline
            ex_filename = self.path + self.base_filename + "exceedences.nc"
        
        for sim in range(0, self.N_simulations):
            # tmax 
            tmp_exceedence_indices_Tmax = np.where(self.T_dailymax[sim, :] >
                                                   baseline_Tmax +
                                                   increment) # gives indexes of exceedences 
            tmp_N_exceedences_Tmax = np.shape(tmp_exceedence_indices_Tmax)[1]
            tmp_percent_exceedences_Tmax = tmp_N_exceedences_Tmax * self.N_days**(-1) * 100
            self.Tmax_exceedences[sim] = tmp_percent_exceedences_Tmax

            # t daily
            tmp_exceedence_indices_tdaily = np.where(self.T_dailymean[sim, :] >
                                                     baseline_Tdaily +
                                                     increment) # gives indexes of exceedences 
            tmp_N_exceedences_tdaily = np.shape(tmp_exceedence_indices_tdaily)[1]
            tmp_percent_exceedences_tdaily = tmp_N_exceedences_tdaily * self.N_days**(-1) * 100
            self.Tdaily_exceedences[sim] = tmp_percent_exceedences_tdaily

            # m daily 
            tmp_exceedence_indices_mdaily = np.where(self.m_dailymean[sim, :] < baseline_mdaily) # gives indexes of exceedences 
            tmp_N_exceedences_mdaily = np.shape(tmp_exceedence_indices_mdaily)[1]
            tmp_percent_exceedences_mdaily = tmp_N_exceedences_mdaily * self.N_days**(-1) * 100
            self.mdaily_exceedences[sim] = tmp_percent_exceedences_mdaily

        print("Finished!")
    
        if save_output == True:
            print("Saving exceedences product...")
            exceedences_ds = xr.Dataset(data_vars={"mean_Z": (["dewpt_T"],
                                                              self.Z_means),
                                                   "Tmax_ex": (["dewpt_T"],
                                                               self.Tmax_exceedences),
                                                   "Tdaily_ex": (["dewpt_T"],
                                                                 self.Tdaily_exceedences),
                                                   "mdaily_ex": (["dewpt_T"],
                                                                 self.mdaily_exceedences),
                                                  },
                                        coords={"dewpt_T": (["dewpt_T"],
                                                            self.Td_means),}
                                       )

            exceedences_ds.to_netcdf(path=ex_filename, mode="w",
                                     format="NETCDF4", engine="netcdf4")

            print("Done!")

    def makeDailyMaximums(self):
        """
        Makes an array of daily maximums for temperature simulation.

        Output:
            - T_dailymax: an array of daily maximum temperatures, for each simulation.
        """
        print("Calculating the daily maximum temperatures in our simulation...")
        for simulation in range(0, self.N_simulations):
            for day in range(0, self.N_days):
                self.T_dailymax[simulation, day] = np.max(self.T_ts[simulation, int(day * self.s_in_day):int((day + 1) * self.s_in_day)]) # take maximum from day's worth of points
        print("Finnished!")

    def makeDailyMeans(self):
        """
        Makes an array of daily means for temperature and soil moisture simulations.

        Output:
            - T_dailymean: an array of daily mean temperatures, for each simulation.
            - m_dailymean: an array of daily mean soil moisture, for each simulation.
        """
        print("Calculating the daily mean temperature and soil moisture...")
        for simulation in range(0, self.N_simulations):
            for day in range(0, self.N_days):
                self.T_dailymean[simulation, day] = np.mean(self.T_ts[simulation, int(day * self.s_in_day):int((day + 1) * self.s_in_day)]) # take maximum from day's worth of points
                self.m_dailymean[simulation, day] = np.mean(self.m_ts[simulation, int(day * self.s_in_day):int((day + 1) * self.s_in_day)])
        print("Finished!")
