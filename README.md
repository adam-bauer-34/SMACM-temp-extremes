# How does soil moisture impact heat wave frequency and intensity over land?

Written by: Adam Michael Bauer

Affiliation: Department of Physics, University of Illinois at Urbana Champaign

To contact: adammb4 [at] illinois [dot] edu

## Contents:
1. [Code description](#codedes)
2. [Publication](#pubs)
3. [Content](#content)

## Code description <a name=“#codedes”></a>
This code integrates the model equations from the Soil Moisture Atmosphere Coupled Model (coined SMACM) in a variety of locations and calculates "exceedences," as well as percentiles and means, of soil moisture and temperature. "Exceedence" data is defined as the percent of days which exceed a specified baseline. In our case, we're interested in how often a baseline for extreme heating events is exceeded, as well as a baseline that defines an extreme drying event is failed to be exceeded. This code enables the user to test how frequent heat waves will be in the future, and how intense they will be. For more information, see the paper or reach out at the email above -- I'd be happy to talk more about this work! 

## Publication <a name=“pubs”></a>
Publication using this code is in preparation. Check back soon!

## Content of repo <a name=“content”></a>
In the top directory is the main file, `HeatwaveFreq_main.py`. This is what a user would run to begin the simulation. In order to change which simulations are done, one must edit the `desired_runs` array in `HeatwaveFreq_main.py`.

### data
The only file in this directory is `BVZP_research_runs.csv`. This file contains information about the various simulations that get ran in `HeatwaveFreq_main.py`. To add more runs, this file must be edited. 

Any data made by the simulation will be saved to this directory.

### notebooks
The analysis notebook for this code is contained here. The notebook comes prepared to be edited easily to make plots analyzing the data. Such plots are exceedence data against the climatological *Z* parameter, the change in mean/max temperature as a function of *Z* for each simulation, as well as daily mean/max temperature histograms and estimated PDFs using KDE methods. 

### src
`LocSimulation.py` is where most of the computations occur. It contains a fairly large class object `LocSimulation` which is called by `HeatwaveFreq_main.py`. `tools.py` contains a variety of I/O functions. 

#### locations
Contains the abstract location class `Location.py`, from which each subclass is sourced (such as `SGP.py`). Each of these files contains information on a location that was analyzed in our paper, such as key model parameter vaues and methods for location-specific simulation calibration.
