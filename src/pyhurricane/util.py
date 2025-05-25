##### import modules

import logging
import re

import numpy as np
import scipy.interpolate
import xarray as xr

from pathlib import Path

#### Logger setting part
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s %(name)s %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

#####
def create_dir(dir_path:str) -> None:
    """
    This function creates a directory if it does not exist.

    Args:
        dir_path (str): The path of the directory to be created.
    """

    path = Path(dir_path)
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"ディレクトリ '{dir_path}' を作成しました。")
    else:
        logging.info(f"ディレクトリ '{dir_path}' は既に存在します。")

#####
def extract_number_before_m(text_in:str) -> int:
    """
    This function tries to extract a numerical value from a string, specifically targeting the number that
    immediately precedes the first occurrence of the character "m".

    Args:
        text_in (str): File name or string containing a number followed by "m".

    Returns:
        int: The extracted number as an integer if found.
        None: If no number is found or if an error occurs.
    """

    try:
        # Use regex to find the number before "m"
        match = re.search(r"(\d+)m", text_in)
        if match:
            return int(match.group(1))
    except Exception as e:
        logging.error(f"Error extracting number: {e}")
        return None

#####
def get_value_time(nc_file_path_in:str) -> tuple:
    """_
    This function reads a NetCDF file and extracts the time variable and its shape.
    It returns the time variable and its shape.

    Args:
        nc_file_path_in (str): Path to the NetCDF file.

    Returns:
        tuple: A tuple containing the time variable and its shape.
    """

    nc_data = xr.open_dataset(nc_file_path_in)       # Read NetCDF file
    time_var = nc_data.time                          # Get time
    [tmt_out] = time_var.shape                       # Get dimesion from time
    time_t_out = time_var[:]                         # Get time data
    nc_data.close()                                  # Close NetCDF file

    return time_t_out, tmt_out

#####
def find_min_mslp_xy(mslp_tyx_in:xr.DataArray) -> tuple:
    """
    This function finds the coordinates (x, y) of the minimum Mean Sea Level Pressure (MSLP) values

    Args:
        mslp_tyx_in (xr.DataArray): Input data array containing MSLP values.
        The first dimension is time, and the second and third dimensions are y and x coordinates.

    Returns:
        tuple: Two arrays containing the x and y coordinates of the minimum MSLP values for each time step.
    """

    num_t = mslp_tyx_in.shape[0]
    min_x_t = np.zeros(num_t, dtype=int)
    min_y_t = np.zeros(num_t, dtype=int)

    for t in range(num_t):
        min_index = np.argmin(mslp_tyx_in[t].values)  # NumPy の np.argmin を使用
        min_y_t[t], min_x_t[t] = np.unravel_index(min_index, mslp_tyx_in[t].shape)

    return min_x_t, min_y_t

#####
def interpolate_time_xy(x_t_in:int, y_t_in:int) -> tuple:
    """
    This function performs linear interpolation on the x and y coordinates of a trajectory over time.
    It takes the x and y coordinates as input and returns the interpolated x and y coordinates.

    Args:
        x_t_in (int): x
        y_t_in (int): y

    Returns:
        tuple: Two arrays containing the interpolated x and y coordinates.
    """

    num_t = x_t_in.shape[0]
    original_time = np.linspace(0, num_t-1, num_t)
    interpolated_time = np.linspace(0, num_t-1, (num_t-1)*60+1)
    interp_x = scipy.interpolate.interp1d(original_time, x_t_in, kind='linear')(interpolated_time)
    interp_y = scipy.interpolate.interp1d(original_time, y_t_in, kind='linear')(interpolated_time)

    return interp_x, interp_y


def interpolate_coord_x(Xcoord:np.ndarray, x:np.ndarray, new_x:np.ndarray) -> np.ndarray:
    """_summary_
    This function performs linear interpolation on a given coordinate array (Xcoord) based on the provided x-coordinates (x)
    and new x-coordinates (new_x). It uses the `scipy.interpolate.interp1d` function to create an interpolation function and then applies it to the new x-coordinates.

    Args:
        Xcoord (np.ndarray):_Description of Xcoord  (e.g., x-coordinates [m]).
        x (np.ndarray): Description of x.
        new_x (np.ndarray): Description of new_x.
        The new x-coordinates for which the interpolation will be performed.

    Returns:
        np.ndarray: The interpolated values corresponding to the new x-coordinates.
    """

    interp_func = scipy.interpolate.interp1d(x, Xcoord, kind='linear', fill_value='extrapolate')
    interpolated_Xcoord = interp_func(new_x)

    return interpolated_Xcoord