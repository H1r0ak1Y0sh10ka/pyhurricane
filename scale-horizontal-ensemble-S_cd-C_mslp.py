##########################################################################
#
#  This python program is makeing figure of Shade; Cd, Contour; MSLP by SCALE netcdf output.  
#  
#  Hiroaki YOSHIOKA, yoshioka-hiroaki-sn@ynu.ac.jp
#  Research fellow at Yokohama National University, Typhoon science and technology Research Center
#
#  --Released History--
#  Jan,12,2024 - 1st released ; Making distribution of Cd and MSLP.
#                             ; You can not handle ensemble members(s_lat,e_lat,s_lon,e_lon).
#　　　　　　　　　　　　　　　 ; Latitude and longitude settings are not implemented. 　　　
#  ???,??,???? - 2nd released ;
#
##########################################################################

import os
import shapefile
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import shapely.geometry as sgeom
import metpy.calc as mpcalc
import scipy.ndimage as ndimage
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from cartopy.vector_transform import vector_scalar_to_grid

##########################################################################

# Data set and TC case 
Model_name       = 'scale' # scale only
TC_name          = 'faxai' # 

# Input part; need parameter, file name, file path and so on.
netcdf_dir_path  = '/mnt/data1/model/scale/faxai/fnl/Cd/'
case_names       = ('CTL')
legends          = ('CTL')
s_parameter      = 'cd'
c_parameter      = 'mslp'
n_skiptime       = 6

s_lat            = 0
e_lat            = 0
s_lon            = 0
e_lon            = 0

# Output part;
experiment_name  = 'ydk' # ' ' 
output_dir_path  = './20240110/'
output_parameter = 'S_'+s_parameter+'-C_'+c_parameter # use for sub directry name 
output_fig_name  = Model_name+'-'+TC_name+'-'+experiment_name+'-'+output_parameter+'-'+case_names+'-'
output_fig_type  = 'png'
dpi              = 1000


##### Prepare output directory  #####

output_dir = os.path.join(output_dir_path,output_parameter)
os.makedirs(output_dir, exist_ok=True)

##### Setting def #####

def get_value_time(nc_file_path):
  # Read NetCDF file
  nc_data = xr.open_dataset(nc_file_path)
  # Get time
  time_var = nc_data.time
  
  # Get dimesion from time
  [tmt] = time_var.shape

  time = time_var[:]

  # Close NetCDF file
  nc_data.close()

  return tmt,time
  
  
##### Prepare dimensions ########################################################

# Number of ensemble members
ne = len(case_names)
# Time information
time_var = (get_value_time(netcdf_dir_path+"/"+case_names+"/"+case_names+".pe000000.nc")[1])
nt       = (get_value_time(netcdf_dir_path+"/"+case_names+"/"+case_names+".pe000000.nc")[0])


##### Main ######################################################################

ds = xr.open_dataset(netcdf_dir_path+"/"+case_names+"/"+case_names+".pe000000.nc")
print("Now reading data is "+netcdf_dir_path+"/"+case_names+"/"+case_names+".pe000000.nc")

plt.rcParams["font.size"] = 18

for it in range(0,nt-1,n_skiptime):  # Loop for time

  ##### Prepare plot field

  fig = plt.figure(figsize=[10, 10], constrained_layout=True)
  ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=140.0,
                                               central_latitude=30,
                                               standard_parallels=(30,40)))#,
  #ax.add_feature(land_50m)
  ax.coastlines(resolution='50m', lw=0.5)
  ax.set_title("INIT; "+time_var[0].dt.strftime("%Y-%m-%d %HUTC").values+", VALUE; "+time_var[it].dt.strftime("%Y-%m-%d %HUTC").values)
  #ax.set_extent([s_lon, e_lon, s_lat, e_lat], crs=ccrs.PlateCarree()) # setting draw area


  gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.33, color='k',alpha=0.5)
  gl.right_labels = gl.top_labels = False
  gl.xlabel_style = {'rotation': 20, 'fontsize': 18}
  gl.ylabel_style = {'fontsize': 18}
  gl.xformatter = LONGITUDE_FORMATTER
  gl.yformatter = LATITUDE_FORMATTER
  gl.xlocator = mticker.FixedLocator(np.arange(130,151,5))
  gl.ylocator = mticker.FixedLocator(np.arange(20,41,5))


  ##### Get variavles
  cd   = ds.USER_CD_VALUE.isel(time=it)
  mslp = ds.MSLP.isel(time=it)*0.01


  ##### Setting shade
  slevels = np.arange(0.0008,0.0027,0.0002)
  sdplot = ax.contourf(ds.lon, ds.lat,
                       cd,
                       transform=ccrs.PlateCarree(),
                       levels = slevels, 
                       cmap="coolwarm",
                       zorder=0)
  fig.colorbar(sdplot, ax=ax, shrink=0.75, ticks=np.arange(0.001, 0.0025, 0.0002))
  
  ##### Setting contour
  clevels = np.arange(960,1061,10)
  colors = ['limegreen']
  cnplot  = ax.contour(ds.lon, ds.lat, 
                       mslp,
                       levels=clevels,
                       transform=ccrs.PlateCarree(),
                       linewidths=2,
                       extend='both', 
                       colors=colors,
                       zorder=1 )
  ax.clabel(cnplot, inline=True)
  
  
  ##### Save figure   
  plt.savefig(output_dir+output_fig_name+time_var[it].dt.strftime("%Y%m%d%H").values+'.'+output_fig_type, format=output_fig_type, dpi=dpi)
  plt.close()
  #plt.show()


