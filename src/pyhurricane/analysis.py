##### ----- import modules

import logging
import numpy as np
import pandas as pd
import netCDF4 as nc
import scipy.ndimage as ndimage
#import scipy.interpolate
#from scipy.interpolate import RegularGridInterpolator

import pyhurricane.util as util

##### ----- Logger setting part
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s %(name)s %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

#####
def making_trackdata(model_data : list,
                    search_x_grid : int = 10,
                    search_y_grid : int = 10,
                    method : str = "mslp"
                    ) -> pd.DataFrame:

    """making_trackdata

    This function processes model data to extract track information based on the specified method (default is "mslp" for minimum sea level pressure).

    Args:
        model_data (list): List of model data file paths.
        search_x_grid (int): Number of search x-grids. Defaults to 10.
        search_y_grid (int): Number of search y-grids. Defaults to 10.
        method (str, optional): Search method. Defaults to "mslp".

    Returns:
        pd.DataFrame: DataFrame containing the track data with columns for time, latitude, longitude, and minimum sea level pressure (MSLP).
    """

    ds =  None

    for ne in model_data:
        print(ne)
        ds = nc.Dataset(ne, chunks={'time': 1, 'z': 'auto', 'y': 'auto', 'x': 'auto'})

        time_var, n_time = util.get_value_time(ne)

        if method == 'mslp':
            output_csv_name = 'mslp'
            var_tyx = ds.variables['MSLP']

        max_nx = (var_tyx[0,:,0].shape)[0]
        max_ny = (var_tyx[0,0,:].shape)[0]

        # Prepare variables
        var_t   = np.zeros((n_time),dtype='float32')
        c_lat_t = np.zeros((n_time),dtype='float32')
        c_lon_t = np.zeros((n_time),dtype='float32')
        c_nx_t  = np.zeros((n_time),dtype='int')
        c_ny_t  = np.zeros((n_time),dtype='int')

        print("SCALE Initial     : ")
        print(time_var[0].values)
        print("SCALE End     : ")
        print(time_var[-1].values)

        for tt in range(0,n_time,1):
            var_yx   = ndimage.gaussian_filter(var_tyx[tt,:,:], sigma=[3,3])

            if search_x_grid == 'ALL':
                var_t[tt] = np.min(var_yx)*0.01
                min_index = np.unravel_index(np.argmin(var_yx, axis=None), var_yx.shape)
                c_nx_t[tt] = min_index[0]
                c_ny_t[tt] = min_index[1]

            else:
                if tt == 0 :
                    var_t[tt] = np.min(var_yx)*0.01
                    min_index = np.unravel_index(np.argmin(var_yx, axis=None), var_yx.shape)
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

                    if c_nx_t[tt-1]+search_x_grid  > max_nx :
                        end_x = max_nx
                    else:
                        end_x =c_nx_t[tt-1]+search_x_grid

                    if c_ny_t[tt-1]+search_y_grid > max_ny :
                        end_y = max_ny
                    else:
                        end_y =c_ny_t[tt-1]+search_y_grid

                    var_t[tt] = np.min(var_yx[start_y:end_y,start_x:end_x])*0.01
                    min_index = np.unravel_index(np.argmin(var_yx[start_y:end_y,start_x:end_x],axis=None),var_yx[start_y:end_y,start_x:end_x].shape)
                    min_coord = (min_index[0] + c_ny_t[tt-1] - search_y_grid, min_index[1] + c_nx_t[tt-1] - search_x_grid )

                    c_nx_t[tt] = min_coord[1]
                    c_ny_t[tt] = min_coord[0]

                # Get lat lon
                c_lat_t[tt]  = ds.variables['lat'][c_ny_t[tt],c_nx_t[tt]]
                c_lon_t[tt]  = ds.variables['lon'][c_ny_t[tt],c_nx_t[tt]]

            # writing csv file
            df = pd.DataFrame({"ft"  :range(0,n_time,1),
                            "time":time_var[:],
                            "lat" :c_lat_t[:],
                            "lon" :c_lon_t[:],
                            "ny"  :c_ny_t[:],
                            "nx"  :c_nx_t[:],
                            "mslp":var_t[:]})
            df = df.set_index('ft')

        print(df)
        var_tyx  = None
        min_index = None
        min_coord = None

        ds.close() # Close NetCDF file

    return df