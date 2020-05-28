import numpy as np
import pandas as pd
from datetime import timedelta, date
import datetime

def extract_DJF(time):
    """
    Function that extracts the DJF months of time

    Input:
    time:      List of time with the format of YYYYMMDD [list] or [ndarray] 

    Output:
    time_DJF:  list of days in DJF months

    Example:
    --------
        >>> time_DJF = extract_DJF([19900101,19901201,19901301])
    
    """

    mm = [floor(tt/100)%100 for tt in time]
    mm = np.array(mm)

    id_DJF = np.where((mm == 12) | (mm == 1) | (mm == 2))
    time_DJF    = [time[idtt] for idtt in id_DJF[0]]

    return time_DJF

def yyyymmdd(time, return_type = 'str'):
    """
    change the data format from datetime64 to yyyymmdd
    
    Input:
    time:           the time needed to be convert [datetime64]
    return_type:    the type of the date, 'str' or 'int', default = 'str'

    Output:
    yyyymmdd:       time of the format of yyyymmdd 

    Example:
    --------
        >>> yyyymmdd = yyyymmdd('1979-01-01T12:00:00','int')

    """
    yyyy = pd.to_datetime(time).year
    mm = pd.to_datetime(time).month
    dd = pd.to_datetime(time).day
    date = str(yyyy)+'%02d' % mm + '%02d' % dd

    if return_type == 'str':
        return date
    else:
        return int(date)


def gener_days(start_year, end_year, return_type = 'str'):
    """
    Function that generates dates from start_year to end_year
    
    Input:
       start_year:     the start year [integer]
       end_year:       the end year [integer]
       return_type:    the type of the date, 'str' or 'int', default = 'str'

    Output:
       days:   list of dates from start_year to end_year
       
    Example:
    --------
      >>> days = gener_days(1980,2000)
    """
    
    start_date = date(start_year,1,1)
    end_date = date(end_year,12,31)
    
    datelist = []

    while start_date <= end_date:
        if return_type == 'str':
            datelist.append(start_date.strftime('%Y%m%d'))
        else:
            datelist.append(int(start_date.strftime('%Y%m%d')))
        start_date += datetime.timedelta(days = 1)
    
    return datelist


def calDayAno(var, ref_start_year, ref_end_year):
    """
    Function that calculates the anomaly of var by removing the annual cycle
    
    Input:
       var:                input variable with the shape of (time, lat, lon) [float]
       ref_start_year:     the start year of the reference time period [integer]
       ref_end_year:       the end year of the reference time period [integer]

    Output:
       var_ano:            the anomalies of var removing the annual cycle with 
                           the same shape of input var
       
    Example:
    --------
      >>> var_ano = calDayAno(var, 1980, 2010)

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    CONTINUED with the missing value!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    """
#     change the missing value to the FillValue( the missing values is sometimes the very 
#     large min values or the very large max value)
    

    # var[var == var.min()] = np.nan
    
    time = var.coords['time']
    ref_period = basic.gener_days(ref_start_year,ref_end_year,'int')
    
    id_period = np.where((time <= ref_period[-1]) & (time >= ref_period[0]))
    var_ref = var[id_period[0],:,:]  # have to specify id_period[0] because id_period is a tuple
     
    mod_ref_period = np.array([date % 10000 for date in ref_period])
    mod_time = np.array([date % 10000 for date in time])
    
    var_ano = var
    
    for kt in range(len(time)):
        id_ref = np.where(mod_ref_period == mod_time[kt])
        var_cycle = np.mean(var_ref[id_ref[0],:,:],axis = 0)
        var_ano[kt,:,:] = var[kt,:,:]-var_cycle
        # var_ano[kt,:,:] = np.where(np.isnan(var[kt,:,:]),var[kt,:,:],var[kt,:,:]-var_ref)
    return var_ano




