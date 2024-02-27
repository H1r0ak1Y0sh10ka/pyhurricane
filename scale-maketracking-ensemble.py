#############################################################################################################
#
#  This python program is makeing typhoon track data(csv) from SCALE netcdf output.  
#  
#  Hiroaki YOSHIOKA, yoshioka-hiroaki-sn@ynu.ac.jp
#  Sep.2022-Present ; Research fellow at Yokohama National University, Typhoon science and technology Research Center
#
#  --Released History--
#  09/12/2023 - 1st released ; TC center positions are detected by Minimum MSLP.
#                              You can handle ensemble members(I use for sensitivity experiments.)
#                              My future plan; adjust to WRF and CReSS output.
#  
#  ??/??/???? - 2nd released ;
#
#############################################################################################################


##### import modules #####

import os
import cftime
import netCDF4 as nc
import numpy as np
import pandas as pd
import datetime as dt
import wxparams as wx
import scipy.ndimage as ndimage

##### Input - Enter Constant, Dirctory, Name, Parameter and so on #####

Model_name = 'scale' # 
TC_name    = 'faxai' # lower case letter is better...probably?

# Input data; Directory path and experimental names ####
netcdf_dir_path = '/mnt/data1/model/scale/faxai/fnl/Cd/'
case_names = ('CTL','Cd1030','Cd1020','Cd1015','Cd1012','Cd1011','Cd1130','Cd1330','Cd1530','Cd1730')


# Chose parameter ####
parameter = 'MSLP'  # now MSLP only
search_x_grid = 5   # 'ALL'
search_y_grid = 15  # 


# Output data; Directory path and output name. ####
output_dir_path   = './trackdata/'
output_csv_name   = Model_name+'-'+TC_name
output_csv_type   = '.csv' # now csv file only


##### Prepare output directory  #####

os.makedirs(output_dir_path, exist_ok=True)


##### Setting def #####

def get_value_time(nc_file_path):
  # Read NetCDF file
  nc_data = nc.Dataset(nc_file_path, 'r')

  # Get time and mslp
  time_var = nc_data.variables['time']

  # Get dimesion from time
  [tmt] = time_var.shape

  # Check units of 'time_var' and convert date from double?
  # e.g.: seconds since ????-??-?? 00:00:00 UTC
  if hasattr(time_var, 'units') and 'seconds since' in time_var.units:
    time = nc.num2date(time_var[:], time_var.units)
  else:
    time = time_var[:]

  # Close NetCDF file
  nc_data.close()

  return tmt,time


##### Main Part #####

# Prepare dimensions for ensemble members
n_cases = len(case_names)

for ne in range(0,n_cases,1):
  print("Now reading data is "+netcdf_dir_path+"/"+case_names[ne]+"/"+case_names[ne]+".pe000000.nc")

  # Get time information
  time_var = (get_value_time(netcdf_dir_path+"/"+case_names[ne]+"/"+case_names[ne]+".pe000000.nc")[1])
  n_time = (get_value_time(netcdf_dir_path+"/"+case_names[ne]+"/"+case_names[ne]+".pe000000.nc")[0])

  # Get variables
  nc_data   = nc.Dataset(netcdf_dir_path+"/"+case_names[ne]+"/"+case_names[ne]+".pe000000.nc", 'r')
  mslp_xyt  = nc_data.variables['MSLP']
    
  # Prepare variables
  var_t   = np.zeros((n_time),dtype='float32')
  c_lat_t = np.zeros((n_time),dtype='float32')
  c_lon_t = np.zeros((n_time),dtype='float32')
  c_nx_t  = np.zeros((n_time),dtype='int')
  c_ny_t  = np.zeros((n_time),dtype='int')

  print("SCALE Initial     : ")
  print(time_var[0])
  print("SCALE End     : ")
  print(time_var[len(time_var)-1])

  for tt in range(0,n_time,1):
    var_xy   = ndimage.gaussian_filter(mslp_xyt[tt,:,:], sigma=[3,3])
    
    # Get dimension for TC center by min mslp
    
    if search_x_grid == 'ALL':
      var_t[tt] = np.min(var_xy)*0.01
      min_index = np.unravel_index(np.argmin(var_xy, axis=None), var_xy.shape)
      c_nx_t[tt] = min_index[0]
      c_ny_t[tt] = min_index[1]

    else:
      if tt == 0 :
        var_t[tt] = np.min(var_xy)*0.01
        min_index = np.unravel_index(np.argmin(var_xy, axis=None), var_xy.shape)
        c_nx_t[tt] = min_index[1]
        c_ny_t[tt] = min_index[0]
        
      else:
        if c_nx_t[tt-1]-search_x_grid  < 0 :
          start_x = 0
        else:
          start_x = c_nx_t[tt-1]-search_x_grid 
          
        if c_ny_t[tt-1]-search_y_grid < 0 :
          start_y = 0
        else:
          start_y = c_ny_t[tt-1]-search_y_grid
        
        if c_nx_t[tt-1]+search_x_grid  > 400 :
          end_x = 400
        else:
          end_x =c_nx_t[tt-1]+search_x_grid 
        
        if c_ny_t[tt-1]+search_y_grid > 420 :
          end_y = 420
        else:
          end_y =c_ny_t[tt-1]+search_y_grid
                 
        var_t[tt] = np.min(var_xy[start_y:end_y,start_x:end_x])*0.01
        min_index = np.unravel_index(np.argmin(var_xy[start_y:end_y,start_x:end_x],axis=None),var_xy[start_y:end_y,start_x:end_x].shape)
        min_coord = (min_index[0] + c_ny_t[tt-1] - search_y_grid, min_index[1] + c_nx_t[tt-1] - search_x_grid )

        c_nx_t[tt] = min_coord[1]
        c_ny_t[tt] = min_coord[0]
 
    # Get lat lon
    c_lat_t[tt]  = nc_data.variables['lat'][c_ny_t[tt],c_nx_t[tt]]
    c_lon_t[tt]  = nc_data.variables['lon'][c_ny_t[tt],c_nx_t[tt]]
      
    # writing csv file
    df = pd.DataFrame({"ft"  :range(0,n_time,1),
                       "time":time_var[:],
                       "lat" :c_lat_t[:],
                       "lon" :c_lon_t[:],
                       "ny"  :c_ny_t[:],
                       "nx"  :c_nx_t[:],
                       "mslp":var_t[:]})
    df = df.set_index('ft')
    df.to_csv(output_dir_path+"/"+output_csv_name+"-"+case_names[ne]+"-"+parameter+"-trackdata"+output_csv_type)
    
  print("Now writing trackdata is "+output_dir_path+"/"+output_csv_name+"-"+case_names[ne]+"-"+parameter+"-trackdata"+output_csv_type)
  
  mslp_xyt  = None
  min_index = None
  min_coord = None

  # Close NetCDF file
  nc_data.close()
  
  print("**********************************************") 

print("Finished !!!")
