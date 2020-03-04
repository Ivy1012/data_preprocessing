#!/usr/bin/env python
# coding: utf-8

# # data_preprocessing
# s1. surface temperature  
# a) Calculate the surface temperature anomaly.
# The anomaly is achieved by subtracting the annual cycle from the t2m data. 
# Annual cycle: average of the same month of each year.

# load data
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import datetime

ncf = Dataset('./data/ERA_interim_t2m_197901_201810.nc')
print(ncf.variables.keys())
t2m = ncf.variables['t2m'][:,:,:]
time = ncf.variables['time'][:]
time_units = ncf.variables['time'].units
lat = ncf.variables['latitude'][:]
lon = ncf.variables['longitude'][:]
# print(ncf)
print('============================')
# print(ncf.variables['latitude'])
# print(ncf.variables['longitude'])
print(ncf.variables['time'])
# print(ncf.variables['t2m'])
print('============================')
ncf.close()


# calculate the anomaly by removing the annual_cycle
def remove_annual_cycle(var):
    '''
    This function is to remove the annual cycle of the input variable
    monthly data are needed.
    
    Paras:
    var :: 3-D data with the 'time' as the first axis.
    
    Return:
    anomaly of the input variable in the same format of the input
    '''
    anomaly              =  np.zeros(var.shape)
    for ii in range(0,12): 
        temp                 =  var[ii::12,:,:]
        annual_cycle         =  np.mean(temp,axis = 0)
        anomaly[ii::12,:,:]      =  var[ii::12,:,:] - annual_cycle
    return anomaly

t2m_ano = remove_annual_cycle(t2m)
print('the shape of the anomaly of the t2m is ', t2m_ano.shape)


# b) Extract the DJF

def extract_DJF(var,time,time_units):
    '''
    This function is to extract specific months from the input varibale.
    
    Paras:
    var        :: 3-D data with the 'time' as the first axis.
    time       :: the corresponding time coordinate of the variable.
    time_units :: the units of time in the format of gregorian.
    
    Return:
    variables with the extracted months
    '''
    start_year = int(time_units[-21:-17])
    start_month = int(time_units[-16:-14])
    start_date = int(time_units[-13:-11])
    intervals = time_units.split()[0]
    
    yyyymm		=	np.zeros(len(time), dtype=np.int32)
    start_time	=	datetime.datetime(start_year,start_month,start_date)
    
    for i in range(len(time)): 
#         it can be changed into command lines.
        if intervals == 'hours':    
            time_temp			=	start_time + datetime.timedelta(hours=int(time[i]))
        elif intervals == 'days':
            time_temp			=	start_time + datetime.timedelta(days=int(time[i]))
        else:
            print('ERROR: the intervals of the time units can only be hours or days')
                
        yyyymm[i]			=	int(time_temp.year * 100 + time_temp.month)

    mm				=	yyyymm%100
    #print(np.min(yyyymm_all),np.max(yyyymm_all))

    D_id	=	mm	==	12
    J_id	=	mm	==	1
    F_id	=	mm	==	2
    var_DJF	=	var[D_id|J_id|F_id,:,:]
    yyyymm_DJF	= yyyymm[D_id|J_id|F_id]
    
    return var_DJF, yyyymm_DJF

t2m_DJF, time_DJF = extract_DJF(t2m_ano,time,time_units)
print(t2m_DJF.shape)
print(t2m_DJF[1,:,:])


# c) Calculate the average values of South-east of CHINA (20º-40ºN, 100-125ºE)


def area_mean(var,lat,lon,east,west,south,north):
    '''
    This function is to calculate the average mean of a rectangle area.
    
    Params:
    var                     ::  3-D data with [time, lat, lon]
    lat,lon                 ::  the coordinate of the variable
    east,west,south,north   ::  The four boundry of a rectangle area in the unit of degree.
    
    Return:
    The average values of the rectangle area in the sequence of time.

    '''
    
    latS	=	lat >= south
    latN	=	lat <= north
    lonW	=	lon >= west
    lonE	=	lon <= east
    
    lat_box	=	lat[latS&latN]
    lon_box	=	lon[lonE&lonW]

    var_box_temp	=	var[:,latS&latN,:]
    var_box	=	var_box_temp[:,:,lonE&lonW]
    
    print(var_box.shape)

    # First we need to convert the latitudes to radians
    latr = np.deg2rad(lat_box)
    # Use the cosine of the converted latitudes as weights for the average
    weights = np.cos(lat_box)
    # Assuming the shape of your data array is (nTimes, nLats, nLons)
    # First find the zonal mean SST by averaging along the latitude circles
    
    var_ave_zonal = var_box.mean(axis=2)
    print(var_ave_zonal.shape)

    # Then take the weighted average of those using the weights we calculated earlier
    var_ave = np.average(var_ave_zonal, axis=1, weights=weights)
    return var_ave

t2m_ano_DJF_SouthEast = area_mean(t2m_DJF,lat,lon,125,100,20,40)
print(t2m_ano_DJF_SouthEast)

# save the data
np.savetxt('./data/t2m_ano_DJF_SouthEast.txt',t2m_ano_DJF_SouthEast,fmt = "%.6f")




