"""Location simulation object

Adam Michael Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
3.7.2022

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

import datetime

import numpy as np 
import xarray as xr 

from .SMACM_integration import integrate_SMACM

class LocSimulation:
    """
    class: LocSimulation

    Simulates for a given location the increase in extreme dry events for a warming event
    of +5 K globally.

    args:
    run_name: string
        run identifier string. to be used in filenames of saved outputs.

    location: Location subclass
        Location subclass containing all model parameters, as well as the 
        amount of forcing increase we need to get ~5 K warming.

    N_summers: int
        number of summers to simulate (higher N_summers, better data)

    N_simulations: int
        number of increments 

    import_precip: bool
        are we importing a previously made precip time series? 

    max_warming: float
        how much are we warming at peak? (in Kelvin)
"""

    def __init__(self, run_name, location, N_simulations, N_summers, import_precip, max_warming):
        self.run_name = run_name
        self.loc = location
        self.N_summers = N_summers
        self.N_simulations = N_simulations
        self.import_precip = import_precip
        self.max_warming = max_warming

        # make base filenames and path
        self.path ="/data/" # UNCOMMENT WHEN READY TO PUSH

        current_date = datetime.datetime.now()
        year = str(current_date.year)
        day = str(current_date.day)
        month = str(current_date.month)
        self.base_filename = ''.join([month, '-', day, '-', year, '-',
                                      self.run_name, '-'])

        # number of days in N summers
        self.N_days = self.N_summers * 90 # 90 days in summer 
        self.s_in_day = 86400 # s / day
        self.N_seconds = self.N_days * self.s_in_day # seconds in a summer 
        self.time = np.arange(0, self.N_seconds, 1) # make time in seconds total in simulation
        self.days = np.arange(0, self.N_days, 1)
        
        # if making new precip time series, make it. if not, import it.
        if self.import_precip == False:
            print("Making new precip forcing...")
            self._make_precip_forcing_ts()
        
        else:
            print("Importing precip time series...")
            self._import_precip()

        # make F range
        self._make_F_means()

        # make mean taus and Zs 
        self.taus = self.loc.mu * self.F_means**(-1) * ((self.loc.alpha_s +
                                                         self.loc.alpha_r) *
                                                        (self.loc.nu *
                                                         self.loc.gamma)**(-1)
                                                        + self.loc.m_0 *
                                                        self.loc.L) 
        self.Z_means = self.loc.omega_s**(-1) * self.taus

        # make forcings 
        self._make_daily_forcing_means() # make daily mean forcing distributions for F and T_d 

        # initial conditions for temperature, moisture simulation
        self.ics = np.asarray([290, 0])

         # make daily maximum temp and daily mean t & m 
        self.T_dailymax = np.zeros((self.N_simulations, self.N_days)) 
        self.T_dailymean = np.zeros_like(self.T_dailymax)
        self.m_dailymean = np.zeros_like(self.T_dailymax)

        # make array for exceedences 
        self.Tmax_exceedences_dyn = np.zeros(self.N_simulations)
        self.Tdaily_exceedences_dyn = np.zeros_like(self.Tmax_exceedences_dyn)
        self.Tmax_exceedences_stat = np.zeros_like(self.Tmax_exceedences_dyn)
        self.Tdaily_exceedences_stat = np.zeros_like(self.Tmax_exceedences_dyn)
        self.mdaily_exceedences = np.zeros_like(self.Tmax_exceedences_dyn)

        # make array for percentiles 
        self.Tmax_percentiles = np.zeros_like(self.Tmax_exceedences_dyn)
        self.Tdaily_percentiles = np.zeros_like(self.Tmax_exceedences_dyn)
        self.mdaily_percentiles = np.zeros_like(self.Tmax_exceedences_dyn)

        # make array for meaans 
        self.Tmax_means = np.zeros_like(self.Tmax_exceedences_dyn)
        self.Tdaily_means = np.zeros_like(self.Tmax_exceedences_dyn)
        self.mdaily_means = np.zeros_like(self.Tmax_exceedences_dyn)

        print("Location Simulation object ready!")

    def _make_precip_forcing_ts(self):
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
                self.P_ts[sec] = np.random.gamma(self.loc.p_0, self.loc.p_scale) # select precip event magnitude from gamma distribution 
                freq_tracker += int(np.random.poisson(self.loc.omega_s)) # the next event occurs freq_tracker + ~omega seconds later 
                self.N_events += 1

            sec += 1
        
        print("Saving precipitation time series...")
        precip_ds = xr.Dataset(data_vars={"precip": (["time"], self.P_ts),
                                         },
                               coords={"time": (["time"], self.time),}
                              )
        
        precip_filename = ''.join([self.path, "precip-ts-",
                                   str(self.N_summers), "sum-", self.run_name,
                                   ".nc"])

        precip_ds.to_netcdf(path=precip_filename, mode="w", format="NETCDF4", engine="netcdf4")
        print("Done!")

    def _import_precip(self):
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
        precip_filename = ''.join([self.path, "precip-ts-",
                                   str(self.N_summers), "sum-", self.run_name,
                                   ".nc"])

        precip_ds = xr.open_dataset(precip_filename)

        # extract precip time series
        self.P_ts = precip_ds["precip"].values 
        print("Precip time series successfully imported!")

    def _make_F_means(self):
        """
        Create a list of radiatve forcing means 
        that increment to our warming situation, which is 
        a global radiative forcing of about 10 W m^-2

        Output: 
            - self.F_means: a list of dew point temperatures, incremented by 5 / N_simulations
        """
        print("Getting ready...")
        self.F_means = np.zeros(self.N_simulations)
        increment = self.loc.F_warming_max * (self.N_simulations - 1)**(-1)
        for sim in range(0, self.N_simulations):
            self.F_means[sim] = self.loc.F_mean + sim * increment

    def _make_daily_forcing_means(self):
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
            self.F_dists[simulation, :] = np.random.normal(self.F_means[simulation], self.loc.F_std, self.N_days)
            self.Td_dists[simulation, :] = np.random.normal(self.loc.Td_mean, self.loc.Td_std, self.N_days)

    def make_model_forcings(self):
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

    def make_forced_ts(self, save_output):
        """
        Carry out Newtonian integration of model equations to get time series for 
        soil moisture and temperature.
        
        Arguments:
            - save_output: bool
                Save output?
        Outputs:
            - T_ts: temperature time series 
            - m_ts: moisture time series 
        """
        print("Creating time series... (this could take a bit of time)")
        model_params = np.asarray([self.loc.alpha_s, self.loc.alpha_r, self.loc.L, self.loc.gamma, self.loc.nu, self.loc.m_0, self.loc.C, self.loc.mu], dtype=np.float32)
        self.T_ts, self.m_ts = integrate_SMACM(self.N_seconds, self.ics, self.P_ts, self.F_ts, self.Td_ts, model_params)
        print("Finished!")

        if save_output == True:
            print("Saving forced time series...")
            ts_ds = xr.Dataset(data_vars={"T_ts": (["F", "time"], self.T_ts),
                                          "m_ts": (["F", "time"], self.m_ts)
                                         },
                               coords={"F": (["F"], self.F_means -
                                             self.loc.F_mean),
                                       "time": (["time"], self.time),}
                              )
           
            ts_ds.to_netcdf(path=''.join([self.path,
                                          self.base_filename, "ts.nc"]),
                            mode='w', format="NETCDF4", engine="netcdf4") # UNCOMMENT WHEN READY TO PUSH
            
            print("Done!")

    def make_simulation_output(self, save_output):
        """
        Calculate the percentage of days which exceed the 95th percentile of
        the baseline daily mean and daily max temperature,
        as well as the 95th percentile for each simulation. 

        Args:
            save_output: bool
            Should we save the output? If True, then a netCDF will be made with
            Z_means, Tmax, Tdaily, and mdaily exceedences recorded

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
        self._make_daily_maximums() # make daily maximum array
        self._make_daily_means() # make daily mean arrays

        print("Calculating 95th and 5th percentiles for each simulation...")
        self.Tmax_percentiles = np.percentile(self.T_dailymax, 95, axis=1)
        self.Tdaily_percentiles = np.percentile(self.T_dailymean, 95, axis=1)
        self.mdaily_percentiles = np.percentile(self.m_dailymean, 5, axis=1)

        print("Calculating means...")
        self.Tmax_means = np.mean(self.T_dailymax, axis=1)
        self.Tdaily_means = np.mean(self.T_dailymean, axis=1)
        self.mdaily_means = np.mean(self.m_dailymean, axis=1)

        print("Calculating exceedences...")
        baseline_Tmax = np.percentile(self.T_dailymax[0, :], 95) # make baseline 95th percentile for daily ax temp
        baseline_Tdaily = np.percentile(self.T_dailymean[0, :], 95) # make baseline 95th percentile for daily mean temp
        baseline_mdaily = np.percentile(self.m_dailymean[0, :], 5) # make baseline 5th percentile for daily mean soil moisture

        T_increment = self.max_warming * (self.N_simulations - 1)**(-1)
        ex_filename = ''.join([self.path, self.base_filename, "exc_perc.nc"])
        
        for sim in range(0, self.N_simulations):
            # tmax dynamic
            tmp_exceedence_indices_Tmax_dyn = np.where(self.T_dailymax[sim, :] >
                                                   baseline_Tmax + sim * T_increment) # gives indexes of exceedences 
            tmp_N_exceedences_Tmax_dyn = np.shape(tmp_exceedence_indices_Tmax_dyn)[1]
            tmp_percent_exceedences_Tmax_dyn = tmp_N_exceedences_Tmax_dyn * self.N_days**(-1) * 100
            self.Tmax_exceedences_dyn[sim] = tmp_percent_exceedences_Tmax_dyn

            # t daily dynamic
            tmp_exceedence_indices_tdaily_dyn = np.where(self.T_dailymean[sim, :] >
                                                     baseline_Tdaily + sim * T_increment) # gives indexes of exceedences 
            tmp_N_exceedences_tdaily_dyn = np.shape(tmp_exceedence_indices_tdaily_dyn)[1]
            tmp_percent_exceedences_tdaily_dyn = tmp_N_exceedences_tdaily_dyn * self.N_days**(-1) * 100
            self.Tdaily_exceedences_dyn[sim] = tmp_percent_exceedences_tdaily_dyn

            # tmax static
            tmp_exceedence_indices_Tmax_stat = np.where(self.T_dailymax[sim, :] >
                                                   baseline_Tmax) # gives indexes of exceedences 
            tmp_N_exceedences_Tmax_stat = np.shape(tmp_exceedence_indices_Tmax_stat)[1]
            tmp_percent_exceedences_Tmax_stat = tmp_N_exceedences_Tmax_stat * self.N_days**(-1) * 100
            self.Tmax_exceedences_stat[sim] = tmp_percent_exceedences_Tmax_stat

            # t daily static
            tmp_exceedence_indices_tdaily_stat = np.where(self.T_dailymean[sim, :] >
                                                     baseline_Tdaily) # gives indexes of exceedences 
            tmp_N_exceedences_tdaily_stat = np.shape(tmp_exceedence_indices_tdaily_stat)[1]
            tmp_percent_exceedences_tdaily_stat = tmp_N_exceedences_tdaily_stat * self.N_days**(-1) * 100
            self.Tdaily_exceedences_stat[sim] = tmp_percent_exceedences_tdaily_stat

            # m daily 
            tmp_exceedence_indices_mdaily = np.where(self.m_dailymean[sim, :] < baseline_mdaily) # gives indexes of exceedences 
            tmp_N_exceedences_mdaily = np.shape(tmp_exceedence_indices_mdaily)[1]
            tmp_percent_exceedences_mdaily = tmp_N_exceedences_mdaily * self.N_days**(-1) * 100
            self.mdaily_exceedences[sim] = tmp_percent_exceedences_mdaily

        print("Finished!")
    
        if save_output == True:
            print("Saving exceedences product...")
            exceedences_ds = xr.Dataset(data_vars={"mean_Z": (["F"],
                                                              self.Z_means),
                                                   "Tmax_ex_dyn": (["F"],
                                                               self.Tmax_exceedences_dyn),
                                                   "Tdaily_ex_dyn": (["F"],
                                                                 self.Tdaily_exceedences_dyn),
                                                    "Tmax_ex_stat": (["F"],
                                                               self.Tmax_exceedences_stat),
                                                   "Tdaily_ex_stat": (["F"],
                                                                 self.Tdaily_exceedences_stat),
                                                   "mdaily_ex": (["F"],
                                                                 self.mdaily_exceedences),
                                                    "Tmax_95perc": (["F"],
                                                                 self.Tmax_percentiles),
                                                   "Tdaily_95perc": (["F"],
                                                                 self.Tdaily_percentiles),
                                                   "mdaily_5perc": (["F"],
                                                                 self.mdaily_percentiles),
                                                    "Tmax_mean_sim": (["F"],
                                                                 self.Tmax_means),
                                                   "Tdaily_mean_sim": (["F"],
                                                                 self.Tdaily_means),
                                                   "mdaily_mean_sim": (["F"],
                                                                 self.mdaily_means),
                                                   "Tdaily_mean": (["F",
                                                                    "day"],
                                                                   self.T_dailymean),
                                                   "Tmax_daily": (["F", "day"],
                                                                 self.T_dailymax),
                                                   "mdaily_mean": (["F",
                                                                    "day"],
                                                                   self.m_dailymean),
                                                  },
                                        coords={"F": (["F"],
                                                            self.F_means -
                                                      self.loc.F_mean),
                                                "day": (['day'], self.days),}
                                       )

            exceedences_ds.to_netcdf(path=ex_filename, mode="w", format="NETCDF4", engine="netcdf4")

            print("Done!")

    def _make_daily_maximums(self):
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

    def _make_daily_means(self):
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