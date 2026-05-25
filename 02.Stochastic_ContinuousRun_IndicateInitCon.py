#!/usr/bin/env python
# coding: utf-8

# ### Import packages that would be used


# In[1]:


from turtle import pd
import jax
import numpy as np
import pickle
import xarray
import gc
import pprint

from netCDF4 import Dataset
import time 
from datetime import datetime, timedelta
from netCDF4 import num2date, date2num


# In[2]:

import sys
import os
os.system("export LD_LIBRARY_PATH=/global/homes/c/chenzm/.conda/envs/py_ai2/lib/python3.12/site-packages/nvidia/cuda_runtime:$LD_LIBRARY_PATH")

from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import neuralgcm

### run all operations in full FP32 precision
from jax import config
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
config.update("jax_default_matmul_precision", "highest")

# test whether jax could identify and find out GPU
jax.devices() 


# ### Initialize API Module and load the weights

# In[3]:

# use downloaded weights
model_name = 'models_v1_precip_stochastic_precip_2_8_deg.pkl'
s_DirRead = './Models/'

# open the weights file 
with open( s_DirRead + model_name, 'rb' ) as f:
    ckpt = pickle.load(f)

model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)



# ### Set up time range
# ### Set up source of initialization conditions
# #### From ERA5
# #### From NeuralGCM
# 

# In[4]:


# function to read files
import xarray.backends
import xarray.coding
from contextlib import contextmanager

def read_file(s_path, s_VarName, s_YYYYMMDDHH):
    s_Read = s_path + ' | grep _' + s_VarName + '.l | grep .' + s_YYYYMMDDHH
    command = f"ls -v {s_Read}"
    # print(command)
    # command = f"ls -v {s_path + ' | grep _' + s_VarName + '. | grep .' + s_YYYYMMDDHH + '_'}"
    results = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
    file_list = results.stdout.splitlines()
    print(file_list)

    # file_list = [s_path + '/' + subfile for subfile in file_list]
    file_list = s_path + '/' + file_list[0]
    
    era5_file = xarray.open_dataset(file_list)
    return era5_file

def read_file_(s_path, s_VarName, s_YYYYMMDDHH):
    s_Read = s_path + ' | grep _' + s_VarName + '_ | grep .' + s_YYYYMMDDHH
    command = f"ls -v {s_Read}"
    # command = f"ls -v {s_path + ' | grep _' + s_VarName + '. | grep .' + s_YYYYMMDDHH + '_'}"
    results = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
    file_list = results.stdout.splitlines()
    print(file_list)

    # file_list = [s_path + '/' + subfile for subfile in file_list]
    file_list = s_path + '/' + file_list[0]
    
    era5_file = xarray.open_dataset(file_list)
    return era5_file

# function to combined all variables
## s_path: data directory
## s_YYYY: year 
## s_MM: month as two-character strings
## s_DD: day as two-character strings
## s_HH: hour as two-character strings
## model_NeuralGCM: model file read from NeuralGCM
def read_initialization_ERA5(s_path, s_YYYY, s_MM, s_DD, s_HH, model_NeuralGCM):
    s_YYYYMMDDHH = s_YYYY + s_MM + s_DD + s_HH
    ## var at pressure levels
    ### U
    era5_U = read_file(s_path + s_YYYY + s_MM + '/', 'u', s_YYYYMMDDHH)
    era5_U = era5_U.sel(time=pd.date_range(s_YYYY + '-' + s_MM + '-' + s_DD + ' ' + s_HH + ':00:00'\
                                        , s_YYYY + '-' + s_MM + '-' + s_DD + ' ' + s_HH + ':00:00', freq='h'))
    # nan_check_U = era5_U.isnull().sum()
    # print(nan_check_U)  # View NaNs in the 'U' dataset

    ### V
    era5_V = read_file(s_path + s_YYYY + s_MM + '/', 'v', s_YYYYMMDDHH)
    era5_V = era5_V.sel(time=pd.date_range(s_YYYY + '-' + s_MM + '-' + s_DD + ' ' + s_HH + ':00:00'\
                                        , s_YYYY + '-' + s_MM + '-' + s_DD + ' ' + s_HH + ':00:00', freq='h'))
    # print(era5_V.data_vars)

    ### Z
    era5_Z = read_file(s_path + s_YYYY + s_MM + '/', "z", s_YYYYMMDDHH)
    era5_Z = era5_Z.sel(time=pd.date_range(s_YYYY + '-' + s_MM + '-' + s_DD + ' ' + s_HH + ':00:00'\
                                        , s_YYYY + '-' + s_MM + '-' + s_DD + ' ' + s_HH + ':00:00', freq='h'))
    # print(era5_Z.data_vars)

    ### T
    era5_T = read_file(s_path + s_YYYY + s_MM + '/', "t", s_YYYYMMDDHH)
    era5_T = era5_T.sel(time=pd.date_range(s_YYYY + '-' + s_MM + '-' + s_DD + ' ' + s_HH + ':00:00'\
                                        , s_YYYY + '-' + s_MM + '-' + s_DD + ' ' + s_HH + ':00:00', freq='h'))
    # print(era5_T.data_vars)

    ### Q
    era5_Q = read_file(s_path + s_YYYY + s_MM + '/', "q", s_YYYYMMDDHH)
    era5_Q = era5_Q.sel(time=pd.date_range(s_YYYY + '-' + s_MM + '-' + s_DD + ' ' + s_HH + ':00:00'\
                                        , s_YYYY + '-' + s_MM + '-' + s_DD + ' ' + s_HH + ':00:00', freq='h'))
    # print(era5_Q.data_vars)

    ### CIWC
    era5_CIWC = read_file(s_path + s_YYYY + s_MM + '/', "ciwc", s_YYYYMMDDHH)
    era5_CIWC = era5_CIWC.sel(time=pd.date_range(s_YYYY + '-' + s_MM + '-' + s_DD + ' ' + s_HH + ':00:00'\
                                        , s_YYYY + '-' + s_MM + '-' + s_DD + ' ' + s_HH + ':00:00', freq='h'))
    # print(era5_CIWC.data_vars)

    ### CLWC
    era5_CLWC = read_file(s_path + s_YYYY + s_MM + '/', "clwc", s_YYYYMMDDHH)
    era5_CLWC = era5_CLWC.sel(time=pd.date_range(s_YYYY + '-' + s_MM + '-' + s_DD + ' ' + s_HH + ':00:00'\
                                        , s_YYYY + '-' + s_MM + '-' + s_DD + ' ' + s_HH + ':00:00', freq='h'))
    # print(era5_CLWC.data_vars)

    ### combine all variables 
    ## merge all the data with all needed var
    full_era5 = xarray.merge([era5_U, era5_V, era5_Z, era5_T, era5_Q, era5_CIWC, era5_CLWC])
    del era5_U, era5_V, era5_Z, era5_T, era5_Q, era5_CIWC, era5_CLWC #, era5_precip
    gc.collect()
    # print(full_era5)

    ### rename the var name 
    var_dict = {
        'U':'u_component_of_wind', 
        'V':'v_component_of_wind', 
        'Z':'geopotential', 
        'T':'temperature', 
        'Q':'specific_humidity', 
        'CIWC':'specific_cloud_ice_water_content', 
        'CLWC':'specific_cloud_liquid_water_content',
        # 'tp':'precipitation_cumulative_mean',
    }
    # rename the var names
    full_era5 = full_era5.rename_vars(var_dict)
    # Keep var needed by the model 
    full_era5 = full_era5[model_NeuralGCM.input_variables]

    return full_era5

def days_in_month(year, month):
    # The monthrange() function returns a tuple (weekday of the first day of the month, number of days in the month)
    return calendar.monthrange(year, month)[1]

## Regrid the initial condition or forcing data 
## set up the regridder 
#Other available regridders include BilinearRegridder and NearestRegridder
def my_regridder(full_era5, model_NeuralGCM):

    # from dinosaur import horizontal_interpolation
    # from dinosaur import spherical_harmonic
    # from dinosaur import xarray_utils

    full_era5_grid = spherical_harmonic.Grid(
        latitude_nodes   = full_era5.sizes['latitude'],
        longitude_nodes  = full_era5.sizes['longitude'],
        latitude_spacing = xarray_utils.infer_latitude_spacing(full_era5.latitude),
        longitude_offset = xarray_utils.infer_longitude_offset(full_era5.longitude),
    )

    regridder = horizontal_interpolation.ConservativeRegridder(
        full_era5_grid, model_NeuralGCM.data_coords.horizontal, skipna=True 
    )

    # regrid 
    regridder = xarray_utils.regrid(full_era5, regridder)
    return regridder

## self-define function to read forcing variables needed by NeuralGCM: SST, CI
## NaN value over land in SST will be replaced by skin temperature (SKT)
## NaN value over land in sea ice (CI) will be set as 0
def read_forcing_ERA5(s_path, s_YYYY, s_MM, s_DD, model_NeuralGCM):
    s_YYYYMM     = s_YYYY + s_MM # '202309'
    s_YYYYMMDD   = s_YYYYMM + s_DD

    ## var at surface
    ### CI
    era5_CI_Day = read_file(s_path + s_YYYYMM + '/', "ci", s_YYYYMM + '0100')
    era5_CI_Day['CI'] = era5_CI_Day['CI'].fillna(0)

    ### SSTK
    era5_SSTK_Day = read_file(s_path + s_YYYYMM + '/', "sstk", s_YYYYMM + '0100')

    #### SKT
    era5_SKT_Day = read_file(s_path + s_YYYYMM + '/', "skt", s_YYYYMM + '0100')
    ##### replace Nan in SST with SKT 
    era5_SSTK_Day['SSTK'] = era5_SSTK_Day['SSTK'].fillna(era5_SKT_Day['SKT'])

    ### combine all variables 
    ## merge all the data with all needed var
    full_era5 = xarray.merge([era5_CI_Day, era5_SSTK_Day])
    del era5_CI_Day, era5_SSTK_Day, era5_SKT_Day
    gc.collect()

    ### rename the var name 
    var_dict = {
        'CI':'sea_ice_cover', 
        'SSTK':'sea_surface_temperature',
    }
    # rename the var names
    full_era5  = full_era5.rename_vars(var_dict)
    # Keep var needed by the model 
    full_era5  = full_era5[model_NeuralGCM.forcing_variables]

    return full_era5

## output the output data as NetCDF file
def output_OneMonthPrediction_NeuralGCM(predictions_ds, output_path, s_YYYY, s_MM, s_DD, Source_of_Initial, i_days):
    # from netCDF4 import Dataset
    # import time 
    # from datetime import datetime, timedelta
    # from netCDF4 import num2date, date2num
    # i_days: day length of simulation

    ### output the raw data
    # output_path = '/global/cfs/cdirs/m1867/zmchen/Work/2024/NGCM_Performance/Data/Test/'
    os.system('mkdir -p ' + output_path)
    output_filename = output_path + 'NGCM_' + s_YYYY + '-' + s_MM + '-' + s_DD + '.nc'
    os.system('rm -rf ' + output_filename)

    predictions_ds_transposed = predictions_ds.copy(deep=True)
    for var in predictions_ds.data_vars:
        # print(predictions_ds[var])
        if 'level' in predictions_ds[var].dims:
            predictions_ds_transposed[var] = predictions_ds[var].transpose('time', 'level', 'latitude', 'longitude')
        elif 'surface' in predictions_ds[var].dims:
            predictions_ds_transposed[var] = predictions_ds[var].transpose('time', 'surface', 'latitude', 'longitude')
        else:
            predictions_ds_transposed[var] = predictions_ds[var]
    ### time 
    start = np.datetime64(s_YYYY + '-' + s_MM + '-' + s_DD + 'T00:00:00')
    # end   = start + np.timedelta64(24, 'h')
    end   = np.datetime64(s_YYYY + '-' + s_MM + '-' + str(i_days) + 'T23:00:00')  #start + np.timedelta64(24, 'h') * i_days
    # print(end)
    # timestamps = np.arange(start, end+1, np.timedelta64(1, 'h'))
    i_Hourly = len(predictions_ds_transposed.coords['time']) // 24
    if i_Hourly == i_days:
        ## hourly output data
        timestamps = np.arange(start, end+1, np.timedelta64(1, 'h'))
    else:
        ## daily output data
        timestamps = np.arange(start, end+1, np.timedelta64(1, 'D'))
    # predictions_ds_transposed = predictions_ds_transposed.assign_coords(time=timestamps)
    dt_series = pd.to_datetime(timestamps) #.strftime('%H:%M %Y/%m/%d')
    yyyymmdd  = dt_series.year * 10000 + dt_series.month * 100 + dt_series.day
    fraction  = dt_series.hour / 24 
    formatted_time = yyyymmdd + fraction
    predictions_ds_transposed = predictions_ds_transposed.assign_coords(time=formatted_time)
    predictions_ds_transposed.time.attrs['long_name'] = 'time'
    # predictions_ds_transposed.time.attrs['units'] = 'YYYYMMDD.fraction'
    predictions_ds_transposed.time.attrs['description'] = 'Date in YYYYMMDD.fraction format where fraction = hour/24'
    predictions_ds_transposed.time.attrs['calendar'] = 'gregorian'
    predictions_ds_transposed.time.attrs['axis'] = 'T'
    predictions_ds_transposed.time.attrs['standard_name'] = 'time'

    ### level 
    level_values = np.array([   1,    2,    3,    5,    7,   10,   20,   30,   50,   70,  100,  125,
                                150,  175,  200,  225,  250,  300,  350,  400,  450,  500,  550,  600,
                                650,  700,  750,  775,  800,  825,  850,  875,  900,  925,  950,  975,
                                1000])
    # predictions_ds_transposed = predictions_ds_transposed.assign_coords(level=level_values)
    predictions_ds_transposed.assign_coords(level=level_values)
    #### Add attributes to the 'level' dimension
    #
    predictions_ds_transposed['level'].attrs['long_name'] = "Pressure Level"
    predictions_ds_transposed['level'].attrs['units']     = "hPa"
    predictions_ds_transposed['level'].attrs['standard_name'] = "air_pressure"
    predictions_ds_transposed['level'].attrs['positive']  = "down"
    predictions_ds_transposed['level'].attrs['axis']      = "Z"
    predictions_ds_transposed['level'].attrs['name']      = "level"
    #
    ## lat & lon
    predictions_ds_transposed['latitude'].attrs['units']  = "degrees_north"
    predictions_ds_transposed['longitude'].attrs['units'] = "degrees_east"
    predictions_ds_transposed['latitude'].attrs['long_name'] = "latitude"
    predictions_ds_transposed['longitude'].attrs['long_name']= "longitude"
    #
    ## geopotential
    predictions_ds_transposed['geopotential'].attrs['units'] = "m2/s2"
    predictions_ds_transposed['geopotential'].attrs['long_name'] = "Geopotential"
    #
    ## temperature
    predictions_ds_transposed['temperature'].attrs['units'] = "K"
    predictions_ds_transposed['temperature'].attrs['long_name'] = "Air Temperature"
    predictions_ds_transposed['temperature'].attrs['standard_name'] = "air_temperature"
    ## P_cumulative
    predictions_ds_transposed['precipitation_cumulative_mean'].attrs['units'] = "kg / (meter**2 * hr)"
    predictions_ds_transposed['precipitation_cumulative_mean'].attrs['long_name'] = "Cumulative Precipitation"
    #
    ## u
    predictions_ds_transposed['u_component_of_wind'].attrs['units'] = "m/s"
    predictions_ds_transposed['u_component_of_wind'].attrs['long_name'] = "Zonal Wind Speed"
    ## v
    predictions_ds_transposed['v_component_of_wind'].attrs['units'] = "m/s"
    predictions_ds_transposed['v_component_of_wind'].attrs['long_name'] = "Meridional Wind Speed"
    ## CLWC
    predictions_ds_transposed['specific_cloud_liquid_water_content'].attrs['units'] = "kg kg**-1"
    predictions_ds_transposed['specific_cloud_liquid_water_content'].attrs['long_name'] = "Specific cloud liquid water content"
    predictions_ds_transposed['specific_cloud_liquid_water_content'].attrs['short_name'] = "clwc"
    ## CIWC
    predictions_ds_transposed['specific_cloud_ice_water_content'].attrs['units'] = "kg kg**-1"
    predictions_ds_transposed['specific_cloud_ice_water_content'].attrs['long_name'] = "Specific cloud ice water content"
    predictions_ds_transposed['specific_cloud_ice_water_content'].attrs['short_name'] = "ciwc"
    ## hus
    predictions_ds_transposed['specific_humidity'].attrs['units'] = "kg kg**-1"
    predictions_ds_transposed['specific_humidity'].attrs['long_name'] = "Specific humidity"
    predictions_ds_transposed['specific_humidity'].attrs['short_name'] = "q"

    ## global attrs 
    write_time  = subprocess.run('date', shell=True, stdout=subprocess.PIPE, text=True)
    write_time  = write_time.stdout.splitlines()
    code_info   = subprocess.run('pwd', shell=True, stdout=subprocess.PIPE, text=True)
    code_info   = code_info.stdout.splitlines() 
    code_info   = os.path.abspath(__file__)
    predictions_ds_transposed.attrs['history'] = f"Created on {write_time[0]}  {code_info}"
    predictions_ds_transposed.attrs['description'] = 'free run output on ' + s_YYYY + '-' + s_MM + '-' + s_DD
    predictions_ds_transposed.attrs['source_of_initial_condition'] = Source_of_Initial
    predictions_ds_transposed.attrs['source']  = 'NeuralGCM; Reference: Kochkov, D. et al., 2024 Nature; Yuval et al. 2026 Science Advances'
    predictions_ds_transposed.attrs['contact'] = "Ziming Chen (ziming.chen@pnnl.gov, ziming.chen17@gmail.com)"

    ### save the dataset to a NetCDF file
    encoding = {
        'time': {
            'dtype': 'float64'
        }
    }
    predictions_ds_transposed.to_netcdf(output_filename, encoding=encoding)

    return 1

def PrintVarSummaryForDataset(var):
    print("Attributes of the variable:")
    for attr_name, attr_value in var.attrs.items():
        print(f"{attr_name}: {attr_value}")
    #
    print("\nCoordinates associated with the variable:")
    for attr_name, attr_value in var.coords.items():
        print(f"{attr_name}: {attr_value}")


def Count_NaN_Range(predictions_ds):
    nan_counts = {var: predictions_ds[var].isnull().sum().item() for var in predictions_ds.data_vars}
    # Print the NaN count for each variable
    for var, count in nan_counts.items():
        if var in ['precipitation_cumulative_mean', 'P_minus_E_cumulative', 'evaporation', 'sea_surface_temperature', 'sea_ice_cover']:
            print(f"{var}: {count} NaN values; range: {predictions_ds[var].min().values} to {predictions_ds[var].max().values}")
        elif var == 'sim_time':
            continue
        else :
            ds_tmp = predictions_ds[var].isel(level=30) # 850 hPa
            min_val= ds_tmp.min().values
            max_val= ds_tmp.max().values
            print(f"{var}: {count} NaN values; range at 850 hPa: {min_val} to {max_val}")
        if count > 100:
            print('----------------------------------------------------------------------------------')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print("All output are NaN. Stop!")
            print('----------------------------------------------------------------------------------')
            cc

@contextmanager
def timer(store):
    start = time.time()
    yield
    end = time.time()
    elapsed_time = end - start
    print(f"Elapsed time: {elapsed_time} seconds")
    store['time'] = elapsed_time

# In[5]:


# main 

### Read ERA5 data
import pandas as pd # we need date_range function
import calendar
import subprocess

# set up time range
Sim_Year_Begin = int(sys.argv[sys.argv.index("--Sim_Year_Begin") + 1])
Sim_Year_End   = int(sys.argv[sys.argv.index("--Sim_Year_End") + 1])
Sim_Mon_Begin  = int(sys.argv[sys.argv.index("--Sim_Mon_Begin") + 1])
Sim_Mon_End    = int(sys.argv[sys.argv.index("--Sim_Mon_End") + 1])
Sim_DD_Begin   = int(sys.argv[sys.argv.index("--Sim_DD_Begin") + 1])
Sim_DD_End     = int(sys.argv[sys.argv.index("--Sim_DD_End") + 1])
#
s_Year       = [str(i) for i in range(Sim_Year_Begin, Sim_Year_End + 1)]
print('Year Ranges:')
print(s_Year[0] + ' to ' + s_Year[-1])
if Sim_Mon_Begin < Sim_Mon_End:
    s_MM         = [str(i).zfill(2) for i in range(Sim_Mon_Begin, Sim_Mon_End + 1)]
elif Sim_Mon_Begin == Sim_Mon_End:
    s_MM         = [str(Sim_Mon_Begin).zfill(2)]
else:
    s_MM         = [str(i).zfill(2) for i in range(Sim_Mon_Begin, 12 + 1)] + [str(i).zfill(2) for i in range(1, Sim_Mon_End + 1)]
print('months:')
print(s_MM)

# set up init condition's date (only work at the beginning of the simulation)
s_yr_Init    = sys.argv[sys.argv.index("--s_yr_Init") + 1]
s_mon_Init   = sys.argv[sys.argv.index("--s_mon_Init") + 1]
s_DD_Init    = sys.argv[sys.argv.index("--s_DD_Init") + 1]
# s_yr_Init    = '1979'
# s_mon_Init   = '02'
# s_DD_Init    = '01'

if '--i_UnifWarming' in sys.argv:
    i_UnifWarming= sys.argv[sys.argv.index("--i_UnifWarming") + 1] # 4 # Units: K
    i_UnifWarming= int(i_UnifWarming)
    s_Exp    = 'amip-p' + str(i_UnifWarming) + 'K'

## Data from ERA5 or NGCM output
path = './ERA5/' 
# path = '/pscratch/sd/y/yeliu/MetOcean/ERA5/'
subdirs = ['e5.oper.an.sfc/', 'e5.oper.an.pl/']
full_paths   = [path + subdir for subdir in subdirs] # Concatenate the base path with each subdirectory

print("InitCon: " + s_yr_Init + '-' + s_mon_Init + '-' + s_DD_Init)

full_path_NGCM = './Output_AMIP_InitCon' + s_yr_Init + '' + s_mon_Init + '' + s_DD_Init + '/'
if 's_Exp' in vars():
    full_path_NGCM = './Output_' + s_Exp + '_InitCon' + s_yr_Init + '' + s_mon_Init + '' + s_DD_Init + '/'

s_Source_of_Initial = 'ERA5'    
print('Source of Initial Condition: ' + s_Source_of_Initial)

# In[6]:

time_dict_total = {}
with timer(time_dict_total):
    ## Read forcing data (SST, sea ice) for each time step, and then combine the initial data
    for s_yr in s_Year:
        for s_mon in s_MM:
            i_month     = int(s_mon)
            i_yr        = int(s_yr)
            i_days      = days_in_month(i_yr, i_month)
            # print(i_days)
            print('')
            #
            time_dict_day = {}
            with timer(time_dict_day):
                # for day in range(1, i_days + 1):
                day        = 1
                #
                s_DD_tmp   = str(day).zfill(2)
                print(s_yr + '-' + s_mon + '-' + s_DD_tmp)
                x_Forcing0 = read_forcing_ERA5(full_paths[0], s_yr, s_mon, s_DD_tmp, model)
                x_Forcing0 = my_regridder(x_Forcing0, model)
                # print(x_Forcing0) # (time: 744, longitude: 128, latitude: 64)
                #
                ## daily average
                x_Forcing  = x_Forcing0.resample(time='1D').mean()
                #
                ## extract the forcing data according to the Sim_DD_Begin and Sim_DD_End
                x_Forcing  = x_Forcing.isel(time=slice(Sim_DD_Begin - 1, Sim_DD_End)) # Sim_DD_Begin and Sim_DD_End are 1-based index
                print("Forcing data time range: " + str(x_Forcing.time[0].values) + ' to ' + str(x_Forcing.time[-1].values))
                #
                ## Time length of the simulations
                i_days_Forcing = len(x_Forcing.coords['time'])
                # if i_days != i_days_Forcing:
                #     print(f"Day length in forcing data, {i_days_Forcing}, is unequal to that in actual month, {i_days}")
                #     print("Stop!")
                #     zz
                new_time_coords = pd.date_range(start=s_yr + '-' + s_mon + '-' + s_DD_tmp + 'T00:00:00', periods=i_days_Forcing, freq='D')
                x_Forcing  = x_Forcing.assign_coords(time=new_time_coords)
                #
                ## for AMIP-PxK
                if 's_Exp' in vars() and 'i_UnifWarming' in vars():
                    # print(f"range: {x_Forcing['sea_surface_temperature'].min().values} to {x_Forcing['sea_surface_temperature'].max().values}")
                    x_Forcing['sea_surface_temperature'].values = x_Forcing['sea_surface_temperature'].values + i_UnifWarming
                    # print(f"range: {x_Forcing['sea_surface_temperature'].min().values} to {x_Forcing['sea_surface_temperature'].max().values}")
                #
                #
                if s_mon == s_MM[0] and s_yr == s_Year[0] and s_Source_of_Initial == 'ERA5':
                    ## Read the initial condition at the beginning of each month
                    # if s_Source_of_Initial == 'ERA5':
                    x_InitialCondition0 = read_initialization_ERA5(full_paths[1], s_yr_Init, s_mon_Init, s_DD_Init, '00', model)
                    x_InitialCondition0 = my_regridder(x_InitialCondition0, model)
                    x_InitialCondition = x_InitialCondition0
                    del x_InitialCondition0
                    gc.collect()
                    #
                    ## change the date to xxxx-xx-01 in x_InitialCondition
                    start  = np.datetime64(s_yr + '-' + s_mon + '-' + s_DD_tmp + 'T00:00:00')
                    end    = start + np.timedelta64(1, 'h')
                    timestamps = np.arange(start, end, np.timedelta64(1, 'h'))
                    x_InitialCondition     = x_InitialCondition.assign_coords(time=timestamps)
                else :
                    if 'predictions_ds' in vars():
                        x_InitialCondition  = predictions_ds.isel(time=-1)
                        x_InitialCondition  = x_InitialCondition[['u_component_of_wind', 'v_component_of_wind', 'geopotential', 'temperature'\
                                                            , 'specific_humidity', 'specific_cloud_ice_water_content'\
                                                            , 'specific_cloud_liquid_water_content', 'precipitation_cumulative_mean'
                                                                ]]                    
                        # adjust the format of time stamps
                        start      = np.datetime64(s_yr + '-' + s_mon + '-' + s_DD_tmp + 'T00:00:00')
                        end        = start + np.timedelta64(1, 'h')
                        timestamps = np.arange(start, end, np.timedelta64(1, 'h'))
                        x_InitialCondition  = x_InitialCondition.assign_coords(time=timestamps)
                        # regrid
                        x_InitialCondition  = my_regridder(x_InitialCondition, model)
                        # 
                #
                ## check NaN values in initial condition and forcing data
                print("\nNaN counts in initial condition and forcing data:")
                nan_counts_IC = Count_NaN_Range(x_InitialCondition)
                nan_counts_FC = Count_NaN_Range(x_Forcing)
                #
                # Align coordinates (now safe: IC and forcing both start at T00:00:00)
                x_InitialCondition, x_Forcing = xarray.align(x_InitialCondition, x_Forcing, join='outer')
                ## merge initial condition and forcing fields
                try:
                    full_era5 = xarray.merge([x_InitialCondition, x_Forcing])
                except Exception as e:
                    print(f"Error during merging: {e}")
                    continue
                #
                ## ready to run the NeuralGCM
                # inner_steps: interval to save model outputs, 1 indicates save model outputs once every 24 hours
                # outer_steps:  output or forward forecast steps 
                # inputs阶段的.isel(time=？)：输入用的时次，这里输入第一个时次，后面的时次用于对比预报效果，这里我没改
                t0          = np.datetime64()
                inner_steps = 1 # save model outputs once every inner_steps (1, 2, ..., or 24) hours
                outer_steps = i_days_Forcing * 24 // inner_steps # total of 1 hour
                timedelta   = np.timedelta64(inner_steps, 'h') #* inner_steps
                times       = (np.arange(outer_steps) * inner_steps) # time axis in hours
                # initialize model state
                ######### Ziming Chen 2025/02/17
                # inputs = model.inputs_from_xarray(full_era5.isel(time=0))
                # input_forcings = model.forcings_from_xarray(full_era5)
                ds_init     = full_era5.isel(time=0)
                inputs, forcings = model.data_from_xarray(ds_init)
                encoded     = model.encode(inputs, forcings, jax.random.key(0)) # transforms input data (on pressure levels) into model variables (on sigma levels)
                # decoded = model.decode(encodes, forcings) # convert back from model levels to pressure levels
                #
                assert model.timestep == np.timedelta64(1, 'h')
                advanced    = model.advance(encoded, forcings)
                #
                if 'predictions_ds' in vars():
                    del predictions_ds
                    gc.collect()
                if 'all_forcings' in vars():
                    del all_forcings
                    gc.collect()
                all_forcings = model.forcings_from_xarray(full_era5)
                if 'final_state' in vars(): #day > 1:
                    del encoded
                    gc.collect()
                    encoded = final_state
                    #
                    del final_state
                    gc.collect()
                # final_state, predictions = model.unroll(encoded, all_forcings, steps=outer_steps, timedelta=np.timedelta64(1, 'h'))
                final_state, predictions = model.unroll(encoded, all_forcings, steps=outer_steps, timedelta=timedelta)
                ######### Ziming Chen 2025/02/17
                #
                predictions_ds = model.data_to_xarray(predictions, times=times)
                # print(predictions_ds)
                #
                # check NaN values
                print("\n\nNaN counts in model predictions/outputs:")
                nan_counts     = Count_NaN_Range(predictions_ds)
                #
                ## Precip
                r_Precip0      = predictions_ds['precipitation_cumulative_mean']
                r_Precip       = r_Precip0.copy() # Make a copy of the ori DataArray
                #
                ## Eva
                r_Eva0         = predictions_ds['evaporation']
                r_Eva          = r_Eva0.copy()
                #
                ### convert to JAX array to use JAX functions
                r_Precip_jax   = jax.numpy.array(r_Precip)
                r_Precip0_jax  = jax.numpy.array(r_Precip0)
                r_Eva_jax      = jax.numpy.array(r_Eva)
                r_Eva0_jax     = jax.numpy.array(r_Eva0)
                # print(r_Precip)
                for itime in range(1, r_Precip.shape[0] + 1):
                    # xarray.DataArray is designed to be immutable with respect to its underlying data when directly accessed this way.
                    # r_Precip.isel(time=itime).values = r_Precip0.isel(time=itime).values - r_Precip0.isel(time=itime-1).values # 
                    r_Precip_jax  = r_Precip_jax.at[itime].set(r_Precip0_jax[itime] - r_Precip0_jax[itime - 1])
                    r_Eva_jax     = r_Eva_jax.at[itime].set(r_Eva0_jax[itime] - r_Eva0_jax[itime - 1])
                # ### deal with the precip at the first step
                if 'r_Precip_jax_LastStep' in vars():
                    r_Precip_jax  = r_Precip_jax.at[0].set(r_Precip_jax[0] - r_Precip_jax_LastStep)
                r_Precip_jax_LastStep = r_Precip0_jax[-1]
                if 'r_Eva_jax_LastStep' in vars():
                    r_Eva_jax  = r_Eva_jax.at[0].set(r_Eva_jax[0] - r_Eva_jax_LastStep)
                r_Eva_jax_LastStep = r_Eva0_jax[-1]
                # ### deal with the precip at the first step
                #
                # Convert the units
                # r_Precip_jax      = r_Precip_jax * 24 # orignal units: kg / (m2 * hr)
                r_Precip_jax      = r_Precip_jax * 1000 * 24# original units: m/hr
                r_Eva_jax         = r_Eva_jax * 1000 * 24# original units: m/hr
                # Convert the JAX array back to a numpy array or xarray
                r_Precip          = jax.device_get(r_Precip_jax)
                xr_Precip         = xarray.DataArray(
                    data=r_Precip,
                    dims=r_Precip0.dims, 
                    coords=r_Precip0.coords,
                    name='precipitation'
                )
                #
                predictions_ds['precipitation'] = xr_Precip
                predictions_ds['precipitation'].attrs['units'] = "mm day**-1"
                predictions_ds['precipitation'].attrs['long_name'] = "mean precipitation"
                predictions_ds['precipitation'].attrs['short_name'] = "precip"

                r_Eva             = jax.device_get(r_Eva_jax)
                xr_Eva            = xarray.DataArray(
                    data=r_Eva,
                    dims=r_Eva0.dims, 
                    coords=r_Eva0.coords,
                    name='evaporation'
                )
                predictions_ds['evaporation'] = xr_Eva
                predictions_ds['evaporation'].attrs['units'] = "mm day**-1"
                predictions_ds['evaporation'].attrs['long_name'] = "mean evaporation"
                predictions_ds['evaporation'].attrs['short_name'] = "eva"
                #
                ## output in each day
                # i_Return = output_NeuralGCM(predictions_ds.isel(time=range(1, 24 + 1)), full_path_NGCM, s_yr, s_mon, s_DD_tmp
                #                             , s_Source_of_Initial + ': ' + s_yr_Init + '-' + s_mon_Init + '-' + s_DD_Init
                #                             )
                i_Return = output_OneMonthPrediction_NeuralGCM(predictions_ds, full_path_NGCM, s_yr, s_mon, s_DD_tmp
                                            , s_Source_of_Initial + ': ' + s_yr_Init + '-' + s_mon_Init + '-' + s_DD_Init
                                            , i_days_Forcing)
                ## output the restart files
                # if day == i_days:
                forcing_tmp = model.forcings_from_xarray(full_era5.isel(time=-1))
                restored_final_state = model.decode(final_state, forcing_tmp)
                #
                i_month     = int(s_mon) + 1
                s_yr_tmp    = s_yr
                #
                if i_month > 12:
                    i_month = 1
                    s_yr_tmp= str(i_yr + 1)
                s_mon_tmp   = str(i_month).zfill(2)
            
                print('-------------------------------------------------------------')
                print('')
                print('')
                #
            print(f"Stored elapsed time: {time_dict_day['time']} seconds")
            s_FileName_SimTime = 'sim_time_each_step.txt'
            if os.path.exists(full_path_NGCM + s_FileName_SimTime):
                with open(full_path_NGCM + s_FileName_SimTime, 'a') as file:
                    file.write(f"\n\nElapsed time in {s_yr}-{s_mon}: {time_dict_day['time']} seconds")
            else:
                with open(full_path_NGCM + s_FileName_SimTime, 'w') as file:
                    file.write(f"Elapsed time in {s_yr}-{s_mon}: {time_dict_day['time']} seconds")
            #
print(f"Total elapsed time in this simulation: {time_dict_total['time']} seconds")
# 
# Open the file in write mode ('w'). Next time I can use 'a' mode to append data to the file
s_date              = datetime.now()
s_date              = s_date.strftime("%Y%m%d")
with open(full_path_NGCM + 'sim_time_' + s_date + '.txt', 'w') as file:
    file.write(f"Total elapsed time in this simulation: {time_dict_total['time']} seconds")

exit(0)

# In[18]:
# ### Vis & check the input data
print(outer_steps)
print(predictions_ds)
print(predictions_ds.coords['time'])

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
def PlottingMultipleFigures(d_Plotting, pressure_levels=None):
    d_Plotting_copy = d_Plotting.copy(deep=True)
    #
    if np.min(d_Plotting_copy.coords['longitude'].values) >= 0.:
        d_Plotting_reverse = d_Plotting_copy.copy(deep=True)
        d_gt180            = d_Plotting_copy.sel(longitude=d_Plotting_copy.longitude[d_Plotting_copy.longitude >= 180])
        d_le180            = d_Plotting_copy.sel(longitude=d_Plotting_copy.longitude[d_Plotting_copy.longitude < 180])
        d_Plotting_reverse.loc[dict(longitude=d_Plotting_reverse.longitude[d_Plotting_reverse.longitude >= 180])] = d_le180.values
        d_Plotting_reverse.loc[dict(longitude=d_Plotting_reverse.longitude[d_Plotting_reverse.longitude <  180])] = d_gt180.values
        # print(d_gt180, d_le180)
        del d_Plotting_copy
        gc.collect()
        d_Plotting_copy    = d_Plotting_reverse.copy(deep=True)
    #
    if pressure_levels is None:
        d_Plotting_copy.plot(
            x='longitude', y='latitude', robust=True, aspect = 2, size = 4, subplot_kws={'projection': ccrs.PlateCarree(central_longitude=180)}
        );
        for ax in plt.gcf().axes:
            if hasattr(ax, 'coastlines'):
                ax.coastlines()
    else:
        for ipre in pressure_levels:
            d_Plotting_copy.sel(level=[ipre], method='nearest').plot(
                x='longitude', y='latitude', row='level', robust=True, aspect = 2, size = 2, subplot_kws={'projection': ccrs.PlateCarree(central_longitude=180)}
            );
            for ax in plt.gcf().axes:
                if hasattr(ax, 'coastlines'):
                    ax.coastlines()

data_inner_steps = 1
# Selecting ERA5 targets from exactly the same time slice
# target_trajectory = model.inputs_from_xarray(
#     regridded_and_filled.thin(time=(inner_steps // data_inner_steps)).isel(time=slice(outer_steps))
# )
# target_data_ds = model.data_to_xarray(target_trajectory, times=times)

# combined_ds = xarray.concat([target_data_ds, predictions_ds], 'model')
# combined_ds.coords['model'] = ['ERA5', 'NerualGCM']

test_temperature = predictions_ds.temperature-273.15

test_temperature.sel(level=1000,time=slice(0,216+1,2)).plot(
    x='longitude', y='latitude', row='time', robust=True, aspect=2, size=2
);

test_precipitation = predictions_ds.precipitation

test_precipitation.sel(time=slice(0,216+1,2)).plot(
    x='longitude', y='latitude', row='time', robust=True, aspect=2, size=2
);

# In[19]:


print(full_era5.var)
return_PlottingMultipleFigures = PlottingMultipleFigures(full_era5.sea_surface_temperature.isel(time=0))
return_PlottingMultipleFigures = PlottingMultipleFigures(full_era5.sea_ice_cover.isel(time=0))


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')

# print(x_InitialCondition) #
# print(x_InitialCondition.coords['longitude'].values)

# Select by specific pressure levels
levels = [100, 200, 500, 850, 1000]
return_PlottingMultipleFigures = PlottingMultipleFigures(x_InitialCondition.temperature.isel(time=0), pressure_levels=levels)


# In[10]:


return_PlottingMultipleFigures = PlottingMultipleFigures(x_InitialCondition.geopotential.isel(time=0), pressure_levels=levels)


# In[11]:


return_PlottingMultipleFigures = PlottingMultipleFigures(x_InitialCondition.specific_humidity.isel(time=0), pressure_levels=levels)


# In[12]:


print(x_InitialCondition.var)
return_PlottingMultipleFigures = PlottingMultipleFigures(x_InitialCondition.u_component_of_wind.isel(time=0), pressure_levels=levels)


# In[13]:


print(x_InitialCondition.var)
return_PlottingMultipleFigures = PlottingMultipleFigures(x_InitialCondition.v_component_of_wind.isel(time=0), pressure_levels=levels)


# In[14]:


print(x_InitialCondition.var)
return_PlottingMultipleFigures = PlottingMultipleFigures(x_InitialCondition.specific_cloud_ice_water_content.isel(time=0), pressure_levels=levels)


# In[15]:


print(x_InitialCondition.var)
return_PlottingMultipleFigures = PlottingMultipleFigures(x_InitialCondition.specific_cloud_liquid_water_content.isel(time=0), pressure_levels=levels)

# In[ ]:


# visualize ERA5 vs NerualGCM trajectories 
combined_ds.specific_humidity.sel(level=850, time=slice(2, 24+1, 3)).plot(
    x='longitude', y='latitude', row='time', col='model', robust=True, aspect=2, size=2
)


# In[44]:



# In[ ]:




