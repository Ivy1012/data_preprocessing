# data_preprocessing

==================================================================
 1. surface temperature (monthly data)  
a) Calculate the surface temperature anomaly.
The anomaly is achieved by subtracting the annual cycle from the t2m data.Annual cycle: average of the same month of each year.  
b) Extract the December&January&February(DJF)  
c) Calculate the average values of South-east of CHINA (20º-40ºN, 100-125ºE)

 2. nino3.4 index  
 extract the DJF month from the original data
 
 3. basic.py
 some functions dealing with the climatology data including:
 a) date processing
 b) calculate the anomalies by removing the annual cycle.
 
4. s3_s3_sensity_of_t2m_to_SST.py
 calculate the area-weighted mean of t2m in china
 and the correlations between the mean and the SST in the pacific
 given the different lead 4 seasons(unit)
