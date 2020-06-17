#!/usr/bin/env python
# coding: utf-8


# 计算中国区域气温均值对不同地区海温异常的敏感性分布
# 中国区域的均值序列，和海温的每个格点的序列
# 以季节为时间单位，错开不同的时次求相关
# calculate the area-weighted mean of t2m in china
# and the correlations between the mean and the SST in the pacific
# given the different lead 4 seasons(unit)


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


# load data
# Read SST using the xarray module.
sst_o = xr.open_dataset('./data/ersst.v5.1m.1854-2019.nc')['sst'].squeeze()
t2m_o = xr.open_dataset('./data/air.2m.mon.mean.nc')['air']
t2m = t2m_o[:,::-1,:]  

sst_time = sst_o.isel(time = slice(1128,1992))
t2m_time = t2m.isel(time = slice(0,864))
# print(sst_time)
# print(t2m_time)

def mask(ds, label='land'):
    landsea = xr.open_dataset('./data/landsea.nc')['lsm']
    landsea = landsea[:,::-1,:].squeeze()
    # interpolation of the leandsea data to the ds
    landsea = landsea.interp(latitude=ds.lat.values, longitude=ds.lon.values)
    ds.coords['mask'] = (('lat', 'lon'), landsea.values)
    if label == 'land':
        ds = ds.where(ds.mask < 0.8,np.nan)
    elif label == 'ocean':
        ds = ds.where(ds.mask > 0.2,np.nan)
    return ds

t2m_mk = mask(t2m_time,label = 'ocean')

# select the china area
t2m_CHN = t2m_mk.sel(lat = slice(20,55),lon = slice(105,125))

# calculate the anomalies by removing the annual cycle
t2ma = t2m_CHN.groupby('time.month') - t2m_CHN.groupby('time.month').mean('time')
lat_t2m = t2m_CHN.coords['lat']
lon_t2m = t2m_CHN.coords['lon']
time = np.arange(1948,2019,1)

t2m_DJF = t2ma.sel(time=t2ma['time.season'] == 'DJF')
t2m_y = np.zeros((71,len(t2ma.coords['lat']),len(t2ma.coords['lon'])),np.float)

print(t2m_y.shape)
for ii in range(t2m_y.shape[0]):
    t2m_temp = t2m_DJF[3*ii+2:3*ii+5,:,:]
    t2m_y[ii,:,:] = np.mean(t2m_temp,axis = 0)
    
t2m_xr = xr.DataArray(t2m_y, dims=('time', 'lat', 'lon'), coords={'time': time,'lat':lat_t2m,'lon':lon_t2m})
# print(t2m_xr)

# calculate the area-weighted mean of t2m
cosy = np.cos(np.deg2rad(t2m_xr.coords['lat'].values)).clip(0., 1.)
# pay attention to the difference of np.mean and np.nanmean and the output type
t_ave_zonal = np.mean(t2m_xr,axis=2)    
# Then take the weighted average of those using the weights we calculated earlier
t2m_ave = np.average(t_ave_zonal, axis=1, weights=cosy)
print(t2m_ave)

# calculate the SST data in different seasons
sst_nino = sst_time.sel(lat = slice(-60,60),lon = slice(120,280)) # it can be adjusted
lat_sst = sst_nino.coords['lat']
lon_sst = sst_nino.coords['lon']
ssta = sst_nino.groupby('time.month') - sst_nino.groupby('time.month').mean('time')
# 分别对提前0-4个季节的SST做季节内平均
sst_DJF = ssta.sel(time=ssta['time.season'] == 'DJF')
sst_SON = ssta.sel(time=ssta['time.season'] == 'SON')
sst_JJA = ssta.sel(time=ssta['time.season'] == 'JJA')
sst_MAM = ssta.sel(time=ssta['time.season'] == 'MAM')

sst_y_DJF = np.zeros((71,len(ssta.coords['lat']),len(ssta.coords['lon'])),np.float)
sst_y_SON = np.zeros((71,len(ssta.coords['lat']),len(ssta.coords['lon'])),np.float)
sst_y_JJA = np.zeros((71,len(ssta.coords['lat']),len(ssta.coords['lon'])),np.float)
sst_y_MAM = np.zeros((71,len(ssta.coords['lat']),len(ssta.coords['lon'])),np.float)
print(sst_y_DJF.shape)

# calculate the mean of seasons (i think it can be improved with more concise code)
for ii in range(sst_y_DJF.shape[0]):
    DJF_temp = sst_DJF[3*ii+2:3*ii+5,:,:]
    sst_y_DJF[ii,:,:] = np.mean(DJF_temp,axis = 0)
    
    SON_temp = sst_SON[3*ii:3*ii+3,:,:]
    sst_y_SON[ii,:,:] = np.mean(SON_temp,axis = 0)
    
    JJA_temp = sst_JJA[3*ii:3*ii+3,:,:]
    sst_y_JJA[ii,:,:] = np.mean(JJA_temp,axis = 0)
    
    MAM_temp = sst_MAM[3*ii:3*ii+3,:,:]
    sst_y_MAM[ii,:,:] = np.mean(MAM_temp,axis = 0)

time = np.arange(1948,2019,1)
# print(time)
sst_xr_DJF = xr.DataArray(sst_y_DJF, dims=('time', 'lat', 'lon'), coords={'time': time,'lat':lat_sst,'lon':lon_sst})
sst_xr_SON = xr.DataArray(sst_y_SON, dims=('time', 'lat', 'lon'), coords={'time': time,'lat':lat_sst,'lon':lon_sst})
sst_xr_JJA = xr.DataArray(sst_y_JJA, dims=('time', 'lat', 'lon'), coords={'time': time,'lat':lat_sst,'lon':lon_sst})
sst_xr_MAM = xr.DataArray(sst_y_MAM, dims=('time', 'lat', 'lon'), coords={'time': time,'lat':lat_sst,'lon':lon_sst})


# print(sst_xr_MAM.mean(axis = 0).max())
# print(sst_xr_MAM.mean(axis = 0).min())

# calculate the correlations

def corr_season(var, data, ci = 0.95):
    corr = np.ma.masked_all(data.shape[1:])
    
    for mm in range(corr.shape[0]):
        for nn in range(corr.shape[1]):
            check_nan = data[:,mm,nn]
            if np.any(np.isnan(check_nan)):
                continue
            c,p = stats.pearsonr(var,data[:,mm,nn])
            corr.data[mm,nn] = c # convert trend to per decade
            sig = (p < (1-ci))
            corr.mask[mm,nn] = ~sig
    return corr

corr_DJF = corr_season(t2m_ave, sst_xr_DJF)
corr_SON = corr_season(t2m_ave, sst_xr_SON)
corr_JJA = corr_season(t2m_ave, sst_xr_JJA)
corr_MAM = corr_season(t2m_ave, sst_xr_MAM)


# plot, mind the projection which matters
sns.set_style('white', {'font.family': 'Arial'})
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()

lonlbl = [u'160°E',u'10°W',u'40°W',u'70°W',u'100°W']
latlbl = [u'10°S',u'5°S',u'0°',u'5°N',u'10°N']
fig = plt.figure(figsize=(12,9),dpi=300)
ax0 = fig.add_subplot(221, projection=ccrs.PlateCarree(central_longitude = 180))
plt.subplots_adjust(wspace =0.5, hspace =0.2)# adjust the space of subplots
cs1 = ax0.contour(lon_sst, lat_sst, sst_xr_DJF.mean(axis = 0), np.linspace(-0.02,0.03,6), colors='k',transform=ccrs.PlateCarree())
cs2 = ax0.contourf(lon_sst, lat_sst, corr_DJF.data, np.linspace(-0.5,0.5,11),
                   cmap=plt.cm.RdBu_r, extend='both',transform=ccrs.PlateCarree())
ax0.contourf(lon_sst, lat_sst, corr_DJF.mask.astype('int'), [-0.5,0.5], hatches=['.','none'],
             colors='none', zorder=10,transform=ccrs.PlateCarree())
ax0.clabel(cs1, inline=1, fontsize=8)
ax0.set_extent([120, 280, -60, 60],crs=ccrs.PlateCarree())
ax0.set_xticks(range(120, 281, 30), crs=ccrs.PlateCarree())
ax0.set_yticks(range(-60, 61, 20), crs=ccrs.PlateCarree())
ax0.xaxis.set_major_formatter(lon_formatter)
ax0.yaxis.set_major_formatter(lat_formatter)
ax0.add_feature(cfeat.COASTLINE, edgecolor='#333333')
ax0.set_title(r'The correlation between t2m and SST_DJF ( head = 0 season)')

ax0 = fig.add_subplot(222,projection=ccrs.PlateCarree(central_longitude = 180))
cs1 = ax0.contour(lon_sst, lat_sst, sst_xr_SON.mean(axis = 0), np.linspace(-0.03,0.01,5), colors='k',transform=ccrs.PlateCarree())
cs2 = ax0.contourf(lon_sst, lat_sst, corr_SON.data, np.linspace(-0.5,0.5,11),
                   cmap=plt.cm.RdBu_r, extend='both',transform=ccrs.PlateCarree())
ax0.contourf(lon_sst, lat_sst, corr_SON.mask.astype('int'), [-0.5,0.5], hatches=['.','none'],
             colors='none', zorder=10,transform=ccrs.PlateCarree())
ax0.clabel(cs1, inline=1, fontsize=8)
ax0.set_extent([120, 280, -60, 60],crs=ccrs.PlateCarree())
ax0.set_xticks(range(120, 281, 30), crs=ccrs.PlateCarree())
ax0.set_yticks(range(-60, 61, 20), crs=ccrs.PlateCarree())
ax0.xaxis.set_major_formatter(lon_formatter)
ax0.yaxis.set_major_formatter(lat_formatter)
ax0.add_feature(cfeat.COASTLINE, edgecolor='#333333')
ax0.set_title(r'The correlation between t2m and SST_SON ( head = 1 season)')

ax0 = fig.add_subplot(223,projection=ccrs.PlateCarree(central_longitude = 180))
cs1 = ax0.contour(lon_sst, lat_sst, sst_xr_JJA.mean(axis = 0), np.linspace(-0.04,0.01,5), 
                  colors='k',transform=ccrs.PlateCarree())
cs2 = ax0.contourf(lon_sst, lat_sst, corr_JJA.data, np.linspace(-0.5,0.5,11),
                   cmap=plt.cm.RdBu_r, extend='both',transform=ccrs.PlateCarree())
ax0.contourf(lon_sst, lat_sst, corr_JJA.mask.astype('int'), [-0.5,0.5], hatches=['.','none'],
             colors='none', zorder=10,transform=ccrs.PlateCarree())
ax0.clabel(cs1, inline=1, fontsize=8)
ax0.set_extent([120, 280, -60, 60],crs=ccrs.PlateCarree())
ax0.set_xticks(range(120, 281, 30), crs=ccrs.PlateCarree())
ax0.set_yticks(range(-60, 61, 20), crs=ccrs.PlateCarree())
ax0.xaxis.set_major_formatter(lon_formatter)
ax0.yaxis.set_major_formatter(lat_formatter)
ax0.add_feature(cfeat.COASTLINE, edgecolor='#333333')
ax0.set_title(r'The correlation between t2m and SST_JJA ( head = 2 seasons)')

ax0 = fig.add_subplot(224,projection=ccrs.PlateCarree(central_longitude = 180))
cs1 = ax0.contour(lon_sst, lat_sst, sst_xr_MAM.mean(axis = 0), np.linspace(-0.04,0.001,5), 
                  colors='k',transform=ccrs.PlateCarree())
cs2 = ax0.contourf(lon_sst, lat_sst, corr_MAM.data, np.linspace(-0.5,0.5,11),
                   cmap=plt.cm.RdBu_r, extend='both',transform=ccrs.PlateCarree())
ax0.contourf(lon_sst, lat_sst, corr_MAM.mask.astype('int'), [-0.5,0.5], hatches=['.','none'],
             colors='none', zorder=10,transform=ccrs.PlateCarree())
ax0.clabel(cs1, inline=1, fontsize=8)
ax0.set_extent([120, 280, -60, 60],crs=ccrs.PlateCarree())
ax0.set_xticks(range(120, 281, 30), crs=ccrs.PlateCarree())
ax0.set_yticks(range(-60, 61, 20), crs=ccrs.PlateCarree())
ax0.xaxis.set_major_formatter(lon_formatter)
ax0.yaxis.set_major_formatter(lat_formatter)
ax0.add_feature(cfeat.COASTLINE, edgecolor='#333333')
ax0.set_title(r'The correlation between t2m and SST_MAM ( head = 3 seasons)')

# color bar with new axes
cxa = fig.add_axes([0.11, 0.08, 0.8, 0.01])
kwargs = {'cax': cxa,'orientation': 'horizontal', 'label': 'correlation coefficients'}
plt.colorbar(cs2,**kwargs)
plt.savefig("./pacific.pdf")

