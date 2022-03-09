"""Main file for heatwave frequency experiment 

Adam Michael Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
2.26.2022

The code increments the temperature by 5 K over the current average
and measures what percentage of the daily mean temperature
and the daily mean soil moisture exceed of fail to reach, resp., a baseline. 

To run: python MasterHeatwaveFreq.py
"""

from src.LocSimulation import LocSimulation
from src.tools import import_csv
from src.locations.SGP import SGP
from src.locations.DAL import DAL
from src.locations.ATL import ATL
from src.locations.SEA import SEA
from src.locations.NY import NY
from src.locations.WIT import WIT

"""
Fill in the runs you want from BVZP_research_runs.csv
"""
desired_runs = [0,1,2,3,4,5]

""" 
Import data for runs.
"""
header, descriptions, data = import_csv("BVZP_research_runs", delimiter=',',
                                        header=True, indices=2)

for run in desired_runs:
    run_name = descriptions[run][1]
    print("Carrying out run %i, which corresponds to %s." % (run, run_name))

    """
    loc_string: name of location we're simulating
    max_warming: maximum amount of temperature warming in our simulation
    N_simulations: number of sub-simulations leading up to maximum warming case
    N_summers: number of summers we integrate the model equations for in each
    simulation
    import_precip: have we already made precip for this run? (check data
    directory!)
    """
    loc_string, max_warming, N_simulations, N_summers, import_precip, dt = data[run]

    """
    Cast variables from BVZP_research_runs.csv into proper forms.
    """
    max_warming = float(max_warming)
    N_simulations = int(N_simulations)
    N_summers = int(N_summers)
    dt = int(dt)

    """
    import_precip must be boolean.
    """
    if import_precip == "True":
        import_precip = True
    else:
        import_precip = False

    """
    Make an instance of the class with name loc_string.
    """
    loc_constructor = globals()[loc_string]
    loc = loc_constructor()

    """
    Calibrate warming simulations with max_warming.
    """
    loc.calibrate_warming_simulations(max_warming=max_warming)

    """
    make class instance for location simulation
    """
    LOC = LocSimulation(run_name, loc, N_simulations, N_summers,
                        import_precip=import_precip,
                        max_warming=max_warming, dt=dt)

    """
    make time series forcing
    """
    LOC.make_model_forcings()

    """
    simulate model equations from SMACM
    """
    LOC.make_forced_ts(save_output=False)

    """
    calculate percentile exceedences for location
    """
    LOC.make_simulation_output(save_output=True)