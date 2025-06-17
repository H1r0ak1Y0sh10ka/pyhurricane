##### import modules

import logging
import os
import re
from pathlib import Path

import numpy as np
import scipy.interpolate
import xarray as xr

##### ----- Logger setting part -----
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
def identify_model_from_netcdf(filepath):
    """
    NetCDFファイルのグローバル属性、変数名、およびファイル名パターンに基づいて、
    そのモデルがWRF、SCALE、または不明であるかを判別します。

    Args:
        filepath (str or Path): 判別したいNetCDFファイルのパス。

    Returns:
        str: 判別されたモデル名 ('WRF', 'SCALE', '不明')。
            ファイルが見つからない、または読み込みエラーが発生した場合は'不明'を返します。
    """
    # Pathオブジェクトに変換し、ファイル存在チェック
    p_path = Path(filepath)
    if not p_path.is_file():
        print(f"エラー: ファイルが見つかりません - {filepath}")
        return '不明'

    try:
        with xr.open_dataset(p_path) as ds:
            # --- 1. グローバル属性の確認 (最も信頼性が高い) ---
            global_attrs = {k.lower(): str(v).lower() for k, v in ds.attrs.items()} # 小文字に統一して検索

            # WRF特有の属性キーワード
            # 'model_name', 'title', 'Conventions' などに 'wrf' や 'arw' が含まれるか
            if 'wrf' in global_attrs.get('model_name', '') or \
                'wrf' in global_attrs.get('title', '') or \
                'arw' in global_attrs.get('conventions', ''):
                return 'WRF'

            # SCALE特有の属性キーワード
            # 'source', 'institution' などに 'scale' や 'riken' が含まれるか
            # SCALEはCFコンベンションに準拠していることが多いので、より具体的なキーワードを探す
            if 'scale' in global_attrs.get('source', '') or \
                'riken' in global_attrs.get('institution', '') or \
                'scale-rm' in global_attrs.get('title', ''): # SCALE-RMの場合など
                return 'SCALE'

            # --- 2. 変数名の確認 ---
            variable_names = {var.lower() for var in ds.data_vars.keys()} # 変数名を小文字セットで取得

            # WRF特有の変数名 (WRFの出力でよく見られるもの)
            # XLAT, XLONG があればWRFの可能性が高い
            wrf_specific_vars = {'xlat', 'xlong', 't', 'p', 'u', 'v', 'qvapor', 'ph', 'phb', 'u10', 'v10', 't2'}
            # 少なくとも2つ以上のWRF特有の変数があればWRFと判断
            if len(variable_names.intersection(wrf_specific_vars)) >= 2:
                # 特にXLATとXLONGは強力なヒント
                if 'xlat' in variable_names and 'xlong' in variable_names:
                    return 'WRF'

            # SCALE特有の変数名 (より汎用的な名前が多いが、特定の組み合わせやファイル名と合わせて)
            # QV (比湿), U, V, T (温度), W (鉛直風), Z (高度) など
            scale_common_vars = {'qv', 'u', 'v', 't', 'w', 'z'} # 仮の例。実際のSCALE出力に合わせて調整が必要
            if len(variable_names.intersection(scale_common_vars)) >= 3:
                # この段階ではまだ断定せず、ファイル名パターンも考慮
                pass

            # --- 3. ファイル名のパターンの確認 ---
            # SCALEの出力ファイルは 'history_dXX.pe######.nc' のようなパターンが多いという情報がある
            filename_lower = p_path.name.lower()
            if 'history_d' in filename_lower and '.pe' in filename_lower and '.nc' in filename_lower:
                return 'SCALE' # ファイル名パターンで判断

            # いずれの条件にも当てはまらない場合
            return '不明'

    except Exception as e:
        print(f"ファイル '{filepath}' の読み込みまたは判別中にエラーが発生しました: {e}")
        return '不明'

def extract_case_name_scale(filepath:str) -> str:
    """
    与えられたファイルパスからケース名（例: CTL, Cd1030）を抽出します。
    パスの形式は '/path/to/flow//CASE_NAME/CASE_NAME.peXXXXXX.nc' を想定しています。
    """
    # 1. ファイル名部分を取得 (例: 'CTL.pe000000.nc')
    basename = os.path.basename(filepath)

    # 2. 拡張子と '.peXXXXXX' の部分を除去
    # '.peXXXXXX.nc' が特定のパターンであるため、最初の'.'までを抽出するのが確実
    if '.' in basename:
        # 最初のドットの前までを取得（例: 'CTL' や 'Cd1030'）
        case_name = basename.split('.')[0]
        return case_name
    else:
        # パターンに合致しない場合（エラーハンドリング）
        return None

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
    """
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