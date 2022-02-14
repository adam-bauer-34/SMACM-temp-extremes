"""
Adam M. Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu

This function removes the seasonal cycle for June, July and August.
In principle, this function works for any set of data with a three month seasonal cycle.
(I suppose this is every season, but I'm not sure if atmospheric scientists sometimes 
use multiple months, like March - October or something.)

Anyways, this removes the seasonal cycle from a three month set of data. 

Feed the function with the data as a NUMPY ARRAY, not as a Dataset or Dataframe or anything like that. 

get2DLaggedRegression computes the linear regression in time (where time is first dimension, i.e., axis = 0) 
of two xarray DataArrays. 
"""

import numpy as np 
from scipy.stats import t
import xarray as xr 

def get2DLaggedRegression(x, y, lagx=0, lagy=0):
    """
    Input: Two xr.Datarrays of any dimensions with the first dim being time. 
    Thus the input data could be a 1D time series, or for example, have three 
    dimensions (time,lat,lon). 
    Datasets can be provided in any order, but note that the regression slope 
    and intercept will be calculated for y with respect to x.
    Output: Covariance, correlation, regression slope and intercept, p-value, 
    and standard error on regression between the two datasets along their 
    aligned time dimension.  
    Lag values can be assigned to either of the data, with lagx shifting x, and
    lagy shifting y, with the specified lag amount. 
    """ 
    #1. Ensure that the data are properly alinged to each other. 
    x,y = xr.align(x,y)

    #2. Add lag information if any, and shift the data accordingly
    if lagx!=0:

        # If x lags y by 1, x must be shifted 1 step backwards. 
        # But as the 'zero-th' value is nonexistant, xr assigns it as invalid 
        # (nan). Hence it needs to be dropped
        x   = x.shift(time = -lagx).dropna(dim='time')

        # Next important step is to re-align the two datasets so that y adjusts
        # to the changed coordinates of x
        x,y = xr.align(x,y)

    if lagy!=0:
        y   = y.shift(time = -lagy).dropna(dim='time')
        x,y = xr.align(x,y)

    #3. Compute data length, mean and standard deviation along time axis: 
    n = y.notnull().sum(dim='time')
    xmean = x.mean(axis=0)
    ymean = y.mean(axis=0)
    xstd  = x.std(axis=0)
    ystd  = y.std(axis=0)

    #4. Compute covariance along time axis
    cov   =  np.sum((x - xmean)*(y - ymean), axis=0)/(n)

    #5. Compute correlation along time axis
    cor   = cov/(xstd*ystd)

    #6. Compute regression slope and intercept:
    slope     = cov/(xstd**2)
    intercept = ymean - xmean*slope  

    #7. Compute P-value and standard error
    #Compute t-statistics
    tstats = cor*np.sqrt(n-2)/np.sqrt(1-cor**2)
    stderr = slope/tstats

    pval   = t.sf(tstats, n-2)*2
    pval   = xr.DataArray(pval, dims=cor.dims, coords=cor.coords)

    return slope, intercept, pval, stderr, cov, cor

def getRemove3MSeasonalCycle(data):
    """
    This function returns the anomalies of a dataset where the seasonal cycle
    has been removed. The data must have the shape (time, latitutde,
    longitude). 

    Args:
        data: (time, lat, long)
        data for seasonal cycle being removed.
    """

    # make dimensions 
    tot_time, lat, long = np.shape(data)
    
    # make extra time, just in case there different number of data per month
    extra_time = tot_time % 3
    
    """
    number of june data points, july data sets, etc.
    """
    tot_june = (tot_time - extra_time) // 3
    tot_july = (tot_time - extra_time) // 3
    tot_aug = (tot_time - extra_time) // 3
    
    """
    If extra time is nonzero, add extra months to the trackers above. 
    """
    if extra_time != 0:
        while extra_time != 0:
            if tot_june <= tot_july:
                tot_june += 1
            else:
                tot_july += 1
            extra_time -= 1
            
    """
    Make new data.
    """
    june_data = np.zeros((tot_june, lat, long))
    july_data = np.zeros((tot_july, lat, long))
    aug_data = np.zeros((tot_aug, lat, long))
    
    """
    Fill arrays.
    """
    june_data[0,:,:] = data[0,:,:]
    july_data[0,:,:] = data[1,:,:]
    aug_data[0,:,:] = data[2,:,:]
    
    june_tracker = 1 
    july_tracker = 1
    aug_tracker = 1
    
    for i in range(3, tot_time):
        if i % 3 == 0:
            #print(i, june_tracker)
            june_data[june_tracker,:,:] = data[i,:,:]
            june_tracker += 1
        elif i % 3 == 1:
            #print(i, july_tracker)
            july_data[july_tracker,:,:] = data[i,:,:]
            july_tracker += 1
        elif i % 3 == 2:
            #print(i, aug_tracker)
            aug_data[aug_tracker,:,:] = data[i,:,:]
            aug_tracker += 1
    
    """
    Take nan free mean to subtract off of the data.
    """
    june_mean = np.nanmean(june_data, axis=0)
    july_mean = np.nanmean(july_data, axis=0)
    aug_mean = np.nanmean(aug_data, axis=0)
    
    """
    Substract off monthly mean.
    """
    june_data -= june_mean
    july_data -= july_mean
    aug_data -= aug_mean
   
    """
    Make final data.
    """
    anom_data = np.concatenate((june_data, july_data, aug_data), axis=0)

    return anom_data
