"""
Adam Michael Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
1/31/2022

This code contains a class object which runs a simulation that calculates the 
percent of days which exceed the 95th percentile of the baseline simulation.

The code increments the dew point temperature by 5 K over the current average
and measures what percentage of daily max temperature exceeds the baseline set 
by the current (baseline) 95th percentile. 

Notes (mostly to self):
    DON'T FORGET TO SAVE THE PRECIP TIME SERIES!! 

To run: python MasterHeatwaveFreq.py [loc_string] [N_simulations] [N_summers]
[dynamic_baseline]

where:
    - loc_string: string
    - N_simulaitons: int
    - N_summers: int
    - dynamic_baseline: bool
"""

from src.LocSimulation import LocSimulation
from src.dictModelParams import loc_param_dict
import matplotlib.pyplot as plt 
import numpy as np
import sys 

"""
LOC_STRING: tells you what location we're simulating for.
Hopefully by the time the code is complete, this is the only parameter that
will need to be changed.
"""
loc_string = sys.argv[1]

"""
Specify the number of simulations (i.e., the number of increments between 0 and 5 K global warming)
and the number of summers we want to simulate for (the higher the better, generally). 
"""
N_simulations = int(sys.argv[2])
N_summers = int(sys.argv[3])

"""
import parameter dictionary from dictModelParams.py
"""
loc_param_dict = loc_param_dict[loc_string]

"""
dynamic baseline?
sys.argv only supports strings as arguments, so here we convert the string
"True" to a boolean True to be passed to the LocSimulation class.
"""
dynamic_baseline = sys.argv[4]

if dynamic_baseline == "True":
    dynamic_baseline = True
else:
    dynamic_baseline = False

"""
make class instance for location we have model parameter fits for
"""
LOC = LocSimulation(loc_string, loc_param_dict, N_simulations, N_summers,
                    import_precip=True)

"""
make time series forcing
"""
LOC.makeModelForcings()

"""
simulate model equations from SMACM
"""
LOC.makeForcedTimeSeries()

"""
calculate percentile exceedences for location
"""
LOC.makeExceedences(save_output=True, dynamic_baseline=dynamic_baseline)
